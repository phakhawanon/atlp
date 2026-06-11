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

def label() -> None:
    """
        - Label all tracked, unlabelled datapoints
        - Write the result in the header file

        Warnings:
            - Require a lot of VRAM from Qwen3-VL-2B-Instruct
            - Must use populate_header() first to ensure that all files are tracked
            
        .. todo::
            - arbitrarily choose VLM model from inputs
            - disable the warning while labelling
            - enhance debugging prints
    """
    root_directory = get_root_directory()
    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    print(f"Start labelling with model {model_name} on path {root_directory}")

    # default: Load the model on the available device(s)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype="auto", device_map="auto",
        #attn_implementation="flash_attention_2"
    )

    processor = AutoProcessor.from_pretrained(model_name)

    data = load_data()

    tag_lists = data["tag_list"]

    for datapoint in data["datapoints"]:

        instruction = get_datapoint(datapoint, use_dict=data, labels=["instruction"])["label"]["instruction"]

        if not instruction:

            success = False
            
            while not success:

                custom_prompt = "You are Galbot G1’s instruction labeller. The robot is instructed to perform a certain task. You are given one RGB videos from the robot’s head camera, while the robot is performing the task. You must determine the instruction given to the robot based on the three videos you have received. The task instruction must be a complete sentence. You must also determine the tags for each task. This tag represents the skill the robot needs to perform the task. There may be more than one tags that can represent the task, but please strictly stick to one tag per task unless the task is complex. Please examine and use the list of the current tags first. If the current tags do not describe the task well, create the new tag. The videos that you have given might sometimes be a failed attempt or incomplete tasks from the teleoperation process. You must also identify if the videos you have been given are the failed or incomplete tasks, which are characterized by hand manipulations with no object involved. Output the result as the .json format. Specify instruction: (string), the tags (list of strings), and is_failed: (boolean). Current tag lists: " + str(tag_lists)

                # Messages containing a video url(or a local path) and a text query
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": str(root_directory / datapoint / "camera_front_head_rgb.mp4"),
                                "fps": 2,
                            },
                            # {
                                # "type": "video",
                                # "video": str(root_directory / datapoint / "camera_left_wrist.mp4"),
                                # "fps": 1,
                            # },
                            # {
                                # "type": "video",
                                # "video": str(root_directory / datapoint / "camera_right_wrist.mp4"),
                                # "fps": 1,
                            # },
                            {"type": "text", "text": custom_prompt},
                        ],
                    }
                ]

                # Preparation for inference
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    fps=2
                )
                inputs = inputs.to(model.device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                try:
                    output_json = json.loads(output_text[0])
                    tag_lists += output_json["tags"]
                    modify_datapoint(datapoint, use_dict=data, labels=output_json)
                    # modify_datapoint(datapoint, use_dict=data, is_failed=output_json["is_failed"], is_labelled=True, is_checked=False, instruction=output_json["instruction"], tags=output_json["tags"])
                    # data["datapoints"][datapoint]["is_failed"] = output_json["is_failed"]
                    # data["datapoints"][datapoint]["is_labelled"] = True
                    # data["datapoints"][datapoint]["is_checked"] = False
                    # data["datapoints"][datapoint]["instruction"] = output_json["instruction"]
                    # data["datapoints"][datapoint]["tags"] = output_json["tags"]
                    # data['count_labelled'] += 1
                    # if output_json["is_failed"]: data['count_failed'] += 1
                    success = True
                    print(f"Labelled {datapoint} with {get_datapoint(datapoint, use_dict=data, labels=['instruction', 'tags','is_failed'])}")
                except json.JSONDecodeError:
                    success = False
        else:
            print(f"Skip {datapoint} as it has alreaby been labelled.")

    data["tag_list"] += tag_lists
    data["tag_list"] = list(set(data["tag_list"]))
    write_data(data)
    print("Finish labelling")
