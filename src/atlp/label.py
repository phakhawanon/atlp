from .interface import (
    load_data,
    get_datapoint,
    modify_datapoint,
    write_data,
    get_root_directory,
)

import json
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    AutoModelForCausalLM,
)
from pathlib import Path
import torch
import numpy as np
import av


prompt_introduction = (
    "You are Galbot G1's instruction labeller."
    "The robot is instructed to perform a certain task."   
    "You are given three RGB videos from the robot's head camera, left wrist, and right wrist while the robot is performing the task."   
    "You must determine the instruction given to the robot based on the three videos you have received.\n"        
)
prompt_instruction_outline = (
    "The task instruction must be a complete sentence."
    "Be very specific by including color and position of the object into the instruction"
    "You must also determine the tags for each task."
    "This tag represents the skill the robot needs to perform the task."
    "There may be more than one tags that can represent the task,"
    "but please strictly stick to one tag per task unless the task is complex."
    "Please examine and use the list of the current tags first."
    "If the current tags do not describe the task well, create the new tag.\n"
)

prompt_instruction_outline_strict = (    
    "The task instruction must be a complete sentence."
    "Be very specific by including color and position of the object into the instruction"
    "You must also determine the tag for each task."
    "This tag represents the skill the robot needs to perform the task."
    "Please carefully select only one tag that best describe the task"
    "Please examine and use the tags that are present in the current tags list"
    "Do not create new tags"
    "If the current tags do not describe the task well, output a list with tag ['Failed'] for tags and True for is_failed"
    "The output instruction and tag meaning must be aligned.\n"
)

prompt_instruction_pouring = (
    "For the pouring tag, if the bottle or any object falls over, mark the task as failed"
    "If the robot does not pour anything, also mark it as failed."
    "For all pouring tag, the lid is intentionally closed. Do not count them as fail.\n"
)

prompt_is_failed_outline = (
    "The videos that you have given might sometimes be a failed attempt"   
    "or incomplete tasks from the teleoperation process."
    "You must also identify if the videos you have been given are the failed or incomplete tasks,"
    "which are characterized by hand manipulations with no object involved"
    "or insuccessful manipulations, characterized by objects failling down\n"
)
prompt_deliverables = (
    "Output the result as a plain string in .json format."
    "(Do not use ```json, just output the plain text only)"
    "Specify instruction: (string), the tags (list of strings), and is_failed: (boolean true or false, lowercase).\n"
)


# ---------------------------------------------------------------------------
# Model-family detection helpers
# ---------------------------------------------------------------------------

def _is_internvl(model_name: str) -> bool:
    return "internvl" in model_name.lower()

def _is_paligemma(model_name: str) -> bool:
    return "paligemma" in model_name.lower()

def _is_llava(model_name: str) -> bool:
    return "llava" in model_name.lower()

def _is_phi(model_name: str) -> bool:
    return "phi" in model_name.lower()


# ---------------------------------------------------------------------------
# Model + processor loading
# ---------------------------------------------------------------------------

def _load_model_and_processor(model_name: str):
    """
    Load the appropriate model and processor class for each model family.

    Notes on each family:
        - Qwen3-VL / InternVL3 : AutoModelForImageTextToText handles both.
        - PaliGemma2            : Also covered by AutoModelForImageTextToText.
        - LLaVA-NeXT-Video      : Requires its own *ForConditionalGeneration class
                                  and LlavaNextVideoProcessor.
        - Phi-3.5-Vision        : Uses AutoModelForCausalLM with trust_remote_code
                                  and AutoProcessor.
    """
    if _is_llava(model_name):
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    elif _is_phi(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, num_crops=4
        )
    else:
        # Qwen3-VL, InternVL3, PaliGemma2
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


# ---------------------------------------------------------------------------
# Video frame sampling (needed for models that don't accept raw video files)
# ---------------------------------------------------------------------------

def _sample_frames(video_path: str, num_frames: int = 8) -> np.ndarray:
    """
    Decode a video file and uniformly sample `num_frames` RGB frames.

    Returns:
        np.ndarray of shape (num_frames, H, W, 3), dtype uint8.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 1
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == num_frames:
            break

    container.close()

    # Pad with last frame if the video is shorter than num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.stack(frames)   # (T, H, W, 3)


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _build_messages(
    model_name: str,
    video_paths: list[str],
    prompt: str,
) -> tuple[list, dict]:
    """
    Build the messages list and apply_chat_template kwargs for the given model.

    Returns:
        messages        : list passed to processor.apply_chat_template
        template_kwargs : extra kwargs for apply_chat_template
    """
    if _is_internvl(model_name):
        content = [{"type": "video", "url": path} for path in video_paths]
        content.append({"type": "text", "text": prompt})
        template_kwargs = {"num_frames": 8}

    elif _is_paligemma(model_name):
        # PaliGemma2 does not natively support video; we sample frames and
        # pass them as a sequence of images instead.
        content = []
        for path in video_paths:
            frames = _sample_frames(path, num_frames=4)   # 4 frames per clip
            for frame in frames:
                content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})
        template_kwargs = {}

    elif _is_llava(model_name):
        # LLaVA-NeXT-Video expects raw numpy arrays (T, H, W, 3) under "video"
        content = []
        for path in video_paths:
            frames = _sample_frames(path, num_frames=8)
            content.append({"type": "video", "video": frames})
        content.append({"type": "text", "text": prompt})
        template_kwargs = {}

    elif _is_phi(model_name):
        # Phi-3.5-Vision uses a special <|image_N|> placeholder syntax.
        # Videos are decomposed into frames passed as individual images.
        images = []
        placeholder_text = ""
        img_idx = 1
        for path in video_paths:
            frames = _sample_frames(path, num_frames=4)
            for frame in frames:
                images.append(frame)
                placeholder_text += f"<|image_{img_idx}|>\n"
                img_idx += 1

        full_text = placeholder_text + prompt
        messages = [{"role": "user", "content": full_text}]
        # Return early — Phi uses a non-standard template flow handled in label()
        return messages, {"images": images, "_phi": True}

    else:
        # Qwen3-VL (default)
        content = [
            {"type": "video", "video": path, "fps": 2} for path in video_paths
        ]
        content.append({"type": "text", "text": prompt})
        template_kwargs = {"fps": 4}

    messages = [{"role": "user", "content": content}]
    return messages, template_kwargs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    model,
    processor,
    model_name: str,
    messages: list,
    template_kwargs: dict,
) -> str:
    """
    Run a single forward pass and return the decoded output string.

    Phi-3.5 is handled separately because it uses AutoModelForCausalLM and
    does not support apply_chat_template in the same way.
    """
    # --- Phi-3.5 ---
    if template_kwargs.pop("_phi", False):
        from PIL import Image as PILImage

        images_np = template_kwargs.pop("images", [])
        pil_images = [PILImage.fromarray(f) for f in images_np]

        prompt_text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(prompt_text, pil_images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # --- All other models (Qwen, InternVL, PaliGemma2, LLaVA) ---
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# ---------------------------------------------------------------------------
# Main labelling entry point
# ---------------------------------------------------------------------------

def label(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    fail_verbose: bool = False,
) -> None:
    """
    Label all tracked, unlabelled datapoints and write results to the header file.

    Supported model families
    ------------------------
    - Qwen3-VL       e.g. ``"Qwen/Qwen3-VL-2B-Instruct"``
    - InternVL3      e.g. ``"OpenGVLab/InternVL3-8B-hf"``
    - PaliGemma2     e.g. ``"google/paligemma2-3b-pt-224"``
    - LLaVA-Video    e.g. ``"llava-hf/LLaVA-NeXT-Video-7B-hf"``
    - Phi-3.5-Vision e.g. ``"microsoft/Phi-3.5-vision-instruct"``

    Warnings
    --------
    - Requires substantial VRAM (amount varies by model size).
    - Run ``populate_header()`` first to ensure all files are tracked.
    - PaliGemma2, LLaVA, and Phi-3.5 require ``av`` (PyAV) for frame sampling:
      ``pip install av``.
    - Phi-3.5 additionally requires ``pillow``.
    """
    root_directory = get_root_directory()
    print(f"Start labelling with model {model_name} on path {root_directory}")

    model, processor = _load_model_and_processor(model_name)

    data = load_data()
    tag_lists = data["tag_list"]

    for datapoint in data["datapoints"]:

        instruction = get_datapoint(
            datapoint, use_dict=data, labels=["instruction"]
        )["label"]["instruction"]

        if not instruction:

            success = False

            while not success:

                prompt_current_tag_list = "Current tag lists: " + str(tag_lists)

                custom_prompt = (
                    prompt_introduction +
                    prompt_is_failed_outline +
                    prompt_instruction_outline_strict +
                    prompt_instruction_pouring +
                    prompt_deliverables +
                    prompt_current_tag_list
                )

                video_paths = [
                    str(root_directory / datapoint / "camera_front_head_rgb.mp4"),
                    str(root_directory / datapoint / "camera_left_wrist.mp4"),
                    str(root_directory / datapoint / "camera_right_wrist.mp4"),
                ]

                messages, template_kwargs = _build_messages(
                    model_name, video_paths, custom_prompt
                )

                output_text = _run_inference(
                    model, processor, model_name, messages, template_kwargs
                )

                output_text = (
                    output_text.strip()
                    .lstrip("```json")
                    .lstrip("```")
                    .rstrip("```")
                    .strip()
                )

                try:
                    output_json = json.loads(output_text)
                    tag_lists += output_json["tags"]
                    tag_lists = list(set(tag_lists))
                    modify_datapoint(datapoint, use_dict=data, labels=output_json)
                    success = True
                    print(
                        f"Labelled {datapoint} with "
                        f"{get_datapoint(datapoint, use_dict=data, labels=['instruction', 'tags', 'is_failed'])}"
                    )
                except json.JSONDecodeError:
                    print("Failed attempt, retrying...")
                    if fail_verbose:
                        print(output_text)
                    success = False

        else:
            print(f"Skip {datapoint} as it has already been labelled.")

    data["tag_list"] += tag_lists
    data["tag_list"] = list(set(data["tag_list"]))
    write_data(data)
    print("Finish labelling")
