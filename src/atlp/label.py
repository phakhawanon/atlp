from .interface import (
    load_data,
    get_datapoint,
    modify_datapoint,
    write_data,
    get_root_directory,
)

import json
from transformers import AutoModelForImageTextToText, AutoProcessor
from pathlib import Path


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
    # "(Do not include [] in front of the output string as well)"
    "Specify instruction: (string), the tags (list of strings), and is_failed: (boolean true or false, lowercase).\n"
)


def _is_internvl(model_name: str) -> bool:
    """Check if the model is an InternVL variant."""
    return "internvl" in model_name.lower()


def _build_messages(model_name: str, video_paths: list[str], prompt: str) -> tuple[list, dict]:
    """
    Build the messages list and apply_chat_template kwargs for the given model.

    Returns:
        messages: list to pass to apply_chat_template
        template_kwargs: extra kwargs for apply_chat_template (fps or num_frames)
    """
    if _is_internvl(model_name):
        # InternVL3: uses "url" key for video, num_frames in apply_chat_template
        content = [
            {"type": "video", "url": path} for path in video_paths
        ]
        content.append({"type": "text", "text": prompt})
        template_kwargs = {"num_frames": 8}
    else:
        # Qwen3-VL (default): uses "video" key, fps in apply_chat_template
        content = [
            {"type": "video", "video": path, "fps": 2} for path in video_paths
        ]
        content.append({"type": "text", "text": prompt})
        template_kwargs = {"fps": 4}

    messages = [{"role": "user", "content": content}]
    return messages, template_kwargs


def label(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    fail_verbose: bool = False,
) -> None:
    """
        - Label all tracked, unlabelled datapoints
        - Write the result in the header file

        Supported models:
            - Qwen3-VL family  (e.g. "Qwen/Qwen3-VL-2B-Instruct")
            - InternVL3 family (e.g. "OpenGVLab/InternVL3-8B-hf")

        Warnings:
            - Require a lot of VRAM
            - Must use populate_header() first to ensure that all files are tracked
            
        .. todo::
            - disable the warning while labelling
            - enhance debugging prints
    """
    root_directory = get_root_directory()
    print(f"Start labelling with model {model_name} on path {root_directory}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    data = load_data()
    tag_lists = data["tag_list"]

    for datapoint in data["datapoints"]:

        instruction = get_datapoint(datapoint, use_dict=data, labels=["instruction"])["label"]["instruction"]

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

                # print(prompt_current_tag_list)

                video_paths = [
                    str(root_directory / datapoint / "camera_front_head_rgb.mp4"),
                    str(root_directory / datapoint / "camera_left_wrist.mp4"),
                    str(root_directory / datapoint / "camera_right_wrist.mp4"),
                ]

                messages, template_kwargs = _build_messages(model_name, video_paths, custom_prompt)

                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    **template_kwargs,
                )
                inputs = inputs.to(model.device)

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # if isinstance(output_text, list): output_text = output_text[0]
                # print(type(output_text[0]))
                output_text = output_text[0]
                output_text = output_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                
                try:
                    output_json = json.loads(output_text)
                    tag_lists += output_json["tags"]
                    tag_lists = list(set(tag_lists))
                    modify_datapoint(datapoint, use_dict=data, labels=output_json)
                    success = True
                    print(f"Labelled {datapoint} with {get_datapoint(datapoint, use_dict=data, labels=['instruction', 'tags', 'is_failed'])}")
                except json.JSONDecodeError:
                    print("Failed attempt, retrying...")
                    if fail_verbose: print(output_text)
                    success = False
        else:
            print(f"Skip {datapoint} as it has already been labelled.")

    data["tag_list"] += tag_lists
    data["tag_list"] = list(set(data["tag_list"]))
    write_data(data)
    print("Finish labelling")
