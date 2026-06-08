"""
    Summary of Automatic Task Labelling Pipeline (ATLP) Module

    Currently support automatic task labelling from Galbot G1 teleoperation dataset

    Note:
        Datapoint is a folder inside the dataset that ends with "_record0"

        The datapoint folder should at least contain camera_front_head_rgb.mp4, which is the RGB video output from the Galbot G1 head camera

    Example:
        >>> import atlp
        >>> set_root(root_to_dataset)
        >>> populate_header() # Track the data in the dataset
        >>> list_datapoints() # Print all details of all datapoints
        >>> show_statistics() # See how many unlabelled datapoints are there in the dataset
        >>> label() # Label all unlabelled datapoints
        >>> display_instruction() # Print all instructions of all datapoints

    .. todo::
        - Structure the code so it can be expanded vertically
        - Implement trivial error handling (input type mismatch, etc) for all functions
"""

# Automatic Task Labelling Pipeline (ATLP)
# Assume that nothing can altered the established header.json

import json
from pathlib import Path
import os
import pandas as pd

# Comment these two lines if you don't want to label anything
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

# Leave this empty if using in this directory
# must end with /
root_directory = Path("~/galbotg1_recorded_data/processed/").expanduser()

# The tuple of endings that will be treated as a datapoint
datapoint_ending = ("_record0")

# The list of default tags
default_tag_list = ["Pick and Place", "Pouring"]

header_path = root_directory / "header.json"

# __all__ = [
#     "display_instruction", "label", "list_datapoints", "populate_header",
#     "reset_label", "set_root", "show_statistics", "tag_add", "tag_get",
#     "tag_remove"
# ]

def load_data() -> dict:
    """
        Load header.json as a dictionary

        Returns:
            Dictionary of the header.json

        .. todo::
            - handle no header file
    """
    
    # Load header file to dict
    with open(header_path, "r") as f:

        data = json.load(f)

    return data


def write_data(data) -> None:
    """
        Write data to header.json

        Args:
            data: Dictionary in .json format

        .. todo::
            - handle no header file
            - handle invalid data input
    """
    
    with open(header_path, "w") as f:

        json.dump(data, f, indent=2)
    

def set_root(root: str) -> None:
    """
        Set the root path

        Args:
            root: The path to the root of the dataset

        Note:
            Always use set_root(path_to_root) atleast once before working with the dataset
    """

    global root_directory, header_path
    root_directory = Path(root).expanduser()
    header_path = root_directory / "header.json"


def has_header() -> bool:
    """
        Check if there is a header file (header.json) in the root directory
    
        Returns:
            True if the current root has appropriate header.json, False otherwise.
    """

    # Check if the file exists
    if not Path(header_path).is_file():

        return False

    # Check if the content is .json
    try:

        data = load_data()

    except json.JSONDecodeError:

        return False

    return True
            

def generate_header() -> None:
    """
        Generate empty header file (header.json) if the header file has not been generated yet
    """

    if has_header():

        print("Header file has already existed")

        return

    if os.path.exists(header_path):

        os.remove(header_path)    

    header_content = {
        "count_datapoint": 0,
        "count_labelled": 0,
        "count_failed": 0,
        "count_checked": 0,
        "tag_list": default_tag_list,
        "datapoints": {}
    }

    write_data(header_content)

    print("Empty header file has been generated")


def count_datapoint_from_root() -> int:
    """
        Return number of datapoints inside the root directory

        Returns:
            Number of datapoints currently inside the root
    """
    
    count_datapoint = sum(1 for entry in os.scandir(root_directory) if entry.is_dir() and entry.name.endswith(datapoint_ending))

    return count_datapoint


def populate_header() -> None:
    """
        - Update the header file to track all datapoints inside the root directory
        - tag_list is untounched.
    """

    # Ensure that header file is generated
    generate_header()

    data = load_data()

    count_datapoint = data["count_datapoint"]
    count_labelled = data["count_labelled"]
    count_failed = data["count_failed"]
    count_checked = data["count_checked"]

    deleted_datapoints = []
        
    # Handling deleted datapoints
    for datapoint in data["datapoints"]:

        datapoint_path = root_directory / datapoint

        if not datapoint_path.is_dir():

            count_datapoint -= 1
            if data["datapoints"][datapoint]["is_failed"]: count_failed -= 1
            if data["datapoints"][datapoint]["is_labelled"]: count_labelled -= 1
            if data["datapoints"][datapoint]["is_checked"]: count_checked -= 1
            deleted_datapoints.append(datapoint)

    for datapoint in deleted_datapoints:
                    
        del data["datapoints"][datapoint] # delete the datapoint

    # Handling untracked datapoints
    if count_datapoint != count_datapoint_from_root() or count_datapoint != len(data["datapoints"]):

        for entry in os.scandir(root_directory):

            if entry.is_dir() and entry.name.endswith(datapoint_ending) and entry.name not in data["datapoints"]:

                data["datapoints"][entry.name] = {
                    "instruction": "",
                    "tags": [],
                    "is_failed": False,
                    "is_labelled": False,
                    "is_checked": False
                }
                count_datapoint += 1
                
    data["count_datapoint"] = count_datapoint
    data["count_labelled"] = count_labelled
    data["count_failed"] = count_failed
    data["count_checked"] = count_checked
    
    write_data(data)

def tag_add(new_tag: str | list[str]) -> None:
    """
        Add new tag(s) to the tag_list of the header file

        Args:
            new_tag: Append that string or a list of strings to the tag_list
    """
    
    data = load_data()

    if type(new_tag) == str:

        data["tag_list"].append(new_tag)
        
    elif type(new_tag) == list:

        data["tag_list"] += new_tag
        
    else:
        
        raise TypeError("Wrong type for new_tag, it must be either str or list")

    # Remove duplicate tags
    data["tag_list"] = list(set(data["tag_list"]))

    write_data(data)


def tag_get() -> list:
    """
        Return tag_list of the header file

        Returns:
            List of all tags inside the header.json
    """

    data = load_data()    

    return data["tag_list"]
    

def tag_remove(tag : str) -> None:
    """
        Remove one tag from the tag_list of header file

        Args:
            tag: String of tag to be removed
    """

    data = load_data()

    if tag in data["tag_list"]:
        data["tag_list"].remove(tag)
        write_data(data)


def show_statistics(list_datapoint=False) -> None:
    """
        Print out statistics of the header file
        
        Args:
            list_datapoint: If true, all datapoint names are printed.
    """

    data = load_data()
    print(f"Statistics for dataset {root_directory}")
    print(f"count_datapoint: {data['count_datapoint']}")
    print(f"count_labelled: {data['count_labelled']}")
    print(f"count_failed: {data['count_failed']}")    
    print(f"count_checkedt: {data['count_checked']}")
    print(f"tag_list: {data['tag_list']}")

    if list_datapoint:
        print("List of Datapoints:")
        for datapoint in data["datapoints"]:
            print(datapoint)


def list_datapoints() -> None:
    """
        Print out all details of each datapoint
    """

    data = load_data()
    df = pd.DataFrame(data["datapoints"])
    print(df.T)


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

        if not data["datapoints"][datapoint]["is_labelled"]:

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
                    data["datapoints"][datapoint]["is_failed"] = output_json["is_failed"]
                    data["datapoints"][datapoint]["is_labelled"] = True
                    data["datapoints"][datapoint]["is_checked"] = False
                    data["datapoints"][datapoint]["instruction"] = output_json["instruction"]
                    data["datapoints"][datapoint]["tags"] = output_json["tags"]
                    data['count_labelled'] += 1
                    if output_json["is_failed"]: data['count_failed'] += 1
                    success = True
                    print(f"Labelled {datapoint} with {data['datapoints'][datapoint]}")
                except json.JSONDecodeError:
                    success = False
        else:
            print(f"Skip {datapoint} as it has alreaby been labelled.")

    data["tag_list"] += tag_lists
    data["tag_list"] = list(set(data["tag_list"]))
    write_data(data)
    print("Finish labelling")


def reset_label() -> None:
    """
        Unlabelled all tracked files in header.json

        Warning:
            When called, all datapoint instructions are deleted.
    """

    data = load_data()

    for datapoint in data["datapoints"]:
        data["datapoints"][datapoint]["instruction"] = ""
        data["datapoints"][datapoint]["tags"] = []
        data["datapoints"][datapoint]["is_failed"] = False
        data["datapoints"][datapoint]["is_labelled"] = False
        data["datapoints"][datapoint]["is_checked"] = False

    data["count_labelled"] = 0
    data["count_failed"] = 0
    data["count_checked"] = 0
    write_data(data)
    print("All tracked filed have been set to unlabelled")


def display_instruction() -> None:
    """
        Print instruction for each datapoint
    """

    data = load_data()

    for datapoint in data["datapoints"]:
        print(f"{datapoint} : {data['datapoints'][datapoint]['instruction']}")

if __name__ == "__main__":
    print("This is ATLP module!")
