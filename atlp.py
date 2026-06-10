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
# Extend atlp to cover model:, actual:

import json
from pathlib import Path
import os
import pandas as pd

# Comment these two lines if you don't want to label anything
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

# Leave this empty if using in this directory
# must end with /
root_directory = Path(".").expanduser()

# The tuple of endings that will be treated as a datapoint
datapoint_ending = ("_record0")

# Set to true to enable collecting actual fields
enable_actual_field = False

# The list of default tags
default_tag_list = ["Pick and Place", "Pouring"]

header_path = root_directory / "header.json"

# __all__ = [
#     "display_instruction", "label", "list_datapoints", "populate_header",
#     "reset_label", "set_root", "show_statistics", "tag_add", "tag_get",
#     "tag_remove"
# ]


# Define values of the datapoint
# field_values = {
#     "instruction": [str],
#     "tags": [list],
#     "is_failed": [False, True],
#     "is_labelled": [False, True], # True if this has been labelled by the model
#     "is_checked": [False, True],
#     "motion_smoothness": [int],
#     "motion_is_joint_violate": [None, False, True],
#     "motion_is_self_collide": [None, False, True],

#     "actual_instruction": [str],    
#     "actual_tags": [list],
#     "actual_motion_smoothness": [None, "Smooth", "Rough"],
#     "actual_is_joint_violate": [None, False, True],
#     "actual_is_self_collide": [None, False, True],
#     # "score": [int]
# }

label_field_values = {
    "instruction": [str],
    "tags": [list],
    "is_failed": [None, False, True],
}

vision_field_values = {
    
}

motion_field_values = {
    "smoothness": [int],
    "is_joint_violate": [None, False, True],
    "is_self_collide": [None, False, True],
}

def append_new_fields() -> None:
    """
        Update the header file to include all fields inside field_values for each datapoint
    """

    data = load_data()

    for datapoint in data["datapoints"]:

        modify_datapoint(datapoint, use_dict=data)
        if enable_actual_field: modify_datapoint(datapoint, use_dict=data, modify_actual=True)

    write_data(data)
        
def _get_default_field_value(field: str, field_values: dict):
    """
        Internal function

        Return the default value for the given field from the given field_values dict

        Do nothing if the given field is not in the field_values
    """

    if field not in field_values:

        print(f"Cannot get default value: field {field} is not in {field_values}")

        return

    default_value = field_values[field][0]
    
    if default_value is str: return ""
    elif default_value is list: return []
    elif default_value is int: return -1
    else: return default_value

def _is_valid_field_value(value, field: str, field_values: dict):
    """
        Internal function

        Return True if the given value is the valid value of the given field from the given field_values.

        Return None if the given field is not inside the given field_values
    """

    if field not in field_values:

        return None

    default_value = field_values[field][0]

    if isinstance(default_value, type): return isinstance(value, default_value)
    elif value in field_values[field]: return True
    else: return False

def set_enable_actual_field(_enable_actual_field: bool) -> None:
    """
        Change enable_actual_field
    """
    global enable_actual_field
    if isinstance(_enable_actual_field, bool): enable_actual_field = _enable_actual_field

def get_datapoint(
    datapoint: str,
    from_actual: bool = False,
    use_dict: dict = dict(),
    labels: list = [],
    visions: list = [],
    motions: list = [],
):
    """
        Safely get the data from the specified datapoint
    """
    return_dict = dict()
    return_labels = dict()
    return_visions = dict()
    return_motions = dict()
    
    # Load data
    is_from_header = False

    if len(use_dict) == 0:

        data = load_data()
        is_from_header = True

    else:

        data = use_dict.copy()

    # Check if there is "actual"
    if from_actual and "actual" not in data["datapoints"][datapoint]:

        print(f"Datapoint {datapoint} does not have actual fields")

        return dict()

    elif not from_actual: data_to_retrieve = data["datapoints"][datapoint]["model"]
    else: data_to_retrieve = data["datapoints"][datapoint]["actual"]

    # Get the values
    for label in labels:

        if label in label_field_values: return_labels[label] = data_to_retrieve["label"][label]
        else: print(f"Invalid label field name: Field '{label}' does not exist.")
    
    for vision in visions:

        if vision in vision_field_values: return_visions[vision] = data_to_retrieve["vision"][vision]
        else: print(f"Invalid vision field name: Field '{vision}' does not exist.")    
    
    for motion in motions:

        if motion in motion_field_values: return_motions[motion] = data_to_retrieve["motion"][motion]
        else: print(f"Invalid motion field name: Field '{motion}' does not exist.")

    return_dict["label"] = return_labels
    return_dict["vision"] = return_visions
    return_dict["motion"] = return_motions

    return return_dict    

def modify_datapoint(
    datapoint: str,
    use_dict: dict = dict(),
    labels: dict = dict(),
    visions: dict = dict(),
    motions: dict = dict(),
    force_add: bool = False,
    reset_label: bool = False,
    reset_vision: bool = False,
    reset_motion: bool = False,
    reset_all: bool = False,
    modify_actual: bool = False,
) -> dict:
    """
        Add new datapoint or modify existing tracked datapoint either from header.json or the given dict

        Args:
            datapoint:
            false_add:
            use_dict:
            reset_label:
            reset_all:
            options:

        .. todo::
            - update global dataset statistics
            - update this documentation
    """

    if reset_all:
        reset_label = True
        reset_vision = True
        reset_motion = True


    # Load data
    is_from_header = False

    if len(use_dict) == 0:

        data = load_data()
        is_from_header = True

    else:

        data = use_dict.copy()

    # Handle new datapoint
    if datapoint not in data["datapoints"]:

        if force_add:

            main_fields = ["model"]

            if enable_actual_field:

                main_fields.append("actual")
                data["datapoints"][datapoint] = {"model": dict(), "actual": dict()}

            else:

                data["datapoints"][datapoint] = {"model": dict()}

            for main_field in main_fields:

                _data_to_modify = data["datapoints"][datapoint][main_field]
            
                _data_to_modify["label"] = dict()
                _data_to_modify["vision"] = dict()
                _data_to_modify["motion"] = dict()

                for field in label_field_values:

                    _data_to_modify["label"][field] = _get_default_field_value(field, label_field_values)

                for field in vision_field_values:

                    _data_to_modify["vision"][field] = _get_default_field_value(field, vision_field_values)

                for field in motion_field_values:

                    _data_to_modify["motion"][field] = _get_default_field_value(field, motion_field_values)
                
        else:

            print("Cannot add the new datapoint. Use force_add=True to forcibly add this datapoint.")

            return dict()
    
    # Handle enable_actual_field
    if enable_actual_field and modify_actual:

        if "actual" not in data["datapoints"][datapoint]: data["datapoints"][datapoint]["actual"] = {"label": dict(), "vision": dict(), "motion": dict()}
        data_to_modify = data["datapoints"][datapoint]["actual"]

    elif not modify_actual:

        data_to_modify = data["datapoints"][datapoint]["model"]

    else:

        print("Modifying actual fields are not possible: enable_actual_field is set to False")        

        return dict()
        
    # Add values to the datapoint
    for label in labels:

        value = labels[label]
        is_valid_value = _is_valid_field_value(value, label, label_field_values)

        if is_valid_value is None: print(f"Invalid label field name: Field '{label}' does not exist.")
        elif is_valid_value: data_to_modify["label"][label] = value
        else: print(f"Invalid label field value: Field '{label}' does not have value '{labels[label]}'")

    for vision in visions:
        
        value = visions[vision]
        is_valid_value = _is_valid_field_value(value, vision, vision_field_values)

        if is_valid_value is None: print(f"Invalid vision field name: Field '{vision}' does not exist.")
        elif is_valid_value: data_to_modify["vision"][vision] = value
        else: print(f"Invalid vision field value: Field '{vision}' does not have value '{visions[vision]}'")

    for motion in motions:
                
        value = motions[motion]
        is_valid_value = _is_valid_field_value(value, motion, motion_field_values)

        if is_valid_value is None: print(f"Invalid motion field name: Field '{motion}' does not exist.")
        elif is_valid_value: data_to_modify["motion"][motion] = value
        else: print(f"Invalid motion field value: Field '{motion}' does not have value '{motions[motion]}'")

    # Handle missing fields
    # Handle resets
    for field in label_field_values:

        if reset_label or field not in data_to_modify["label"]:

            data_to_modify["label"][field] = _get_default_field_value(field, label_field_values)
    
    for field in vision_field_values:

        if reset_vision or field not in data_to_modify["vision"]:

            data_to_modify["vision"][field] = _get_default_field_value(field, vision_field_values)

    
    for field in motion_field_values:

        if reset_motion or field not in data_to_modify["motion"]:

            data_to_modify["motion"][field] = _get_default_field_value(field, motion_field_values)



    # Handle obsolete fields

    # label
    delete_fields = list()
    
    for data_field in data_to_modify["label"]:

        if data_field not in label_field_values: delete_fields.append(data_field)

    for data_field in delete_fields: del data_to_modify["label"][data_field]

    # vision
    delete_fields = list()
    
    for data_field in data_to_modify["vision"]:

        if data_field not in vision_field_values: delete_fields.append(data_field)

    for data_field in delete_fields: del data_to_modify["vision"][data_field]

    # motion
    delete_fields = list()
    
    for data_field in data_to_modify["motion"]:

        if data_field not in motion_field_values: delete_fields.append(data_field)

    for data_field in delete_fields: del data_to_modify["motion"][data_field]


    # Write data
    if modify_actual: data["datapoints"][datapoint]["actual"] = data_to_modify
    else: data["datapoints"][datapoint]["model"] = data_to_modify

    # Write/return data
    if is_from_header:

        write_data(data)

        return dict()

    else:

        return data    

    # # OBSOLETE Handle new datapoint
    # if datapoint not in data["datapoints"] or reset_label:

    #     if force_add or reset_label:
            
    #         data["datapoints"][datapoint] = dict()

    #         for field in field_values:

    #             if field.startswith("actual_") and not reset_all:

    #                 pass

    #             else:

    #                 if field_values[field][0] is str:

    #                     write_value = ""

    #                 elif field_values[field][0] is list:

    #                     write_value = []

    #                 elif field_values[field][0] is int:

    #                     write_value = 0

    #                 else:

    #                     write_value = field_values[field][0]
        
    #                 data["datapoints"][datapoint][field] = write_value

    #     else:

    #         print("Cannot add the new datapoint. Use force_add=True to forcibly add this datapoint.")

    #         return dict()

    # # OBSOLETE Add values into the specified fields
    # for option in options:

    #     if option in field_values:
        
    #         if options[option] not in field_values[option] and not isinstance(options[option], field_values[option][0]):

    #             print(f"Invalid field value: Field '{option}' does not has value '{options[option]}'")

    #         else:

    #             data["datapoints"][datapoint][option] = options[option]

    #     else:

    #         print(f"Invalid field name: Field '{option}' does not exists.")

    # # OBSOLETE Handle missing fields
    # for field in field_values:

    #     if field not in data["datapoints"][datapoint]:

    #         if field_values[field][0] is str:

    #             write_value = ""

    #         elif field_values[field][0] is list:

    #             write_value = []

    #         elif field_values[field][0] is int:

    #             write_value = 0

    #         else:

    #             write_value = field_values[field][0]
    
    #         data["datapoints"][datapoint][field] = write_value

    # # OBSOLETE Handle obsolete fields
    # delete_fields = list()
    
    # for data_field in data["datapoints"][datapoint]:

    #     if data_field not in field_values:

    #         delete_fields.append(data_field)

    # for data_field in delete_fields:
                
    #     del data["datapoints"][datapoint][data_field]









        
# To be safely retired     
def update_statistics(use_dict: dict=dict()) -> dict:
    """
        Update the statistics of the header.json or the given dict
    """
    
    # Load data
    is_from_header = False

    if len(use_dict) == 0:

        data = load_data()
        is_from_header = True

    else:

        data = use_dict.copy()

    count_datapoint = 0
    # count_labelled = 0
    # count_failed = 0
    # count_checked = 0
    tag_list = data["tag_list"]

    for datapoint in data["datapoints"]:

        count_datapoint += 1

        # if data["datapoints"][datapoint]["is_labelled"]: count_labelled += 1
        # if data["datapoints"][datapoint]["is_failed"]: count_failed += 1
        # if data["datapoints"][datapoint]["is_checked"]: count_checked += 1

        tag_list += data["datapoints"][datapoint]["model"]["label"]["tags"]
        tag_list = list(set(tag_list))

    data["count_datapoint"] = count_datapoint
    # data["count_labelled"] = count_labelled
    # data["count_failed"] = count_failed
    # data["count_checked"] = count_checked
    data["tag_list"] = tag_list
    
    # Write/return data
    if is_from_header:

        write_data(data)

        return dict()

    else:

        return data
   
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
        # "count_labelled": 0,
        # "count_failed": 0,
        # "count_checked": 0,
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

    # update_statistics()
    append_new_fields()

    data = load_data()

    count_datapoint = data["count_datapoint"]
    # count_labelled = data["count_labelled"]
    # count_failed = data["count_failed"]
    # count_checked = data["count_checked"]

    deleted_datapoints = []
        
    # Handling deleted datapoints
    for datapoint in data["datapoints"]:

        datapoint_path = root_directory / datapoint

        if not datapoint_path.is_dir():

            count_datapoint -= 1
            # if data["datapoints"][datapoint]["is_failed"]: count_failed -= 1
            # if data["datapoints"][datapoint]["is_labelled"]: count_labelled -= 1
            # if data["datapoints"][datapoint]["is_checked"]: count_checked -= 1
            deleted_datapoints.append(datapoint)

    for datapoint in deleted_datapoints:
                    
        del data["datapoints"][datapoint] # delete the datapoint

    # Handling untracked datapoints
    if count_datapoint != count_datapoint_from_root() or count_datapoint != len(data["datapoints"]):

        for entry in os.scandir(root_directory):

            if entry.is_dir() and entry.name.endswith(datapoint_ending) and entry.name not in data["datapoints"]:

                # data["datapoints"][entry.name] = {
                #     "instruction": "",
                #     "tags": [],
                #     "is_failed": False,
                #     "is_labelled": False,
                #     "is_checked": False
                # }
                modify_datapoint(entry.name, use_dict=data, force_add=True)
                count_datapoint += 1
                
    data["count_datapoint"] = count_datapoint
    # data["count_labelled"] = count_labelled
    # data["count_failed"] = count_failed
    # data["count_checked"] = count_checked
    
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

        .. todo::
            - Recursively remove the tag from all datapoints in the header file
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
    update_statistics()
    append_new_fields()

    data = load_data()
    print(f"Statistics for dataset {root_directory}")
    print(f"count_datapoint: {data['count_datapoint']}")
    # print(f"count_labelled: {data['count_labelled']}")
    # print(f"count_failed: {data['count_failed']}")    
    # print(f"count_checked: {data['count_checked']}")
    print(f"tag_list: {data['tag_list']}")

    if list_datapoint:
        print("List of Datapoints:")
        for datapoint in data["datapoints"]:
            print(datapoint)


def list_datapoints(from_actual: bool = False) -> None:
    """
        Print out all details of each datapoint

        .. todo::
            - modify or retire this function
    """
    update_statistics()
    append_new_fields()

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


def reset_all(modify_actual: bool = False) -> None:
    """
        Reset every value of every fields of all datapoints to their default values

        Args:
            modify_actual:
                    
        Warning:
            When called, all datapoint information is deleted.
    """

    data = load_data()

    for datapoint in data["datapoints"]:
        modify_datapoint(datapoint, use_dict=data, reset_all=True, modify_actual=modify_actual)
        # data["datapoints"][datapoint]["instruction"] = ""
        # data["datapoints"][datapoint]["tags"] = []
        # data["datapoints"][datapoint]["is_failed"] = False
        # data["datapoints"][datapoint]["is_labelled"] = False
        # data["datapoints"][datapoint]["is_checked"] = False

    # data["count_labelled"] = 0
    # data["count_failed"] = 0
    # data["count_checked"] = 0
    write_data(data)
    print("All tracked filed have been set to unlabelled")


def display_instruction(from_actual: bool = False) -> None:
    """
        Print instruction for each datapoint
    """

    data = load_data()

    for datapoint in data["datapoints"]:

        instruction = get_datapoint(datapoint, from_actual=from_actual, use_dict=data, labels=['instruction'])#['label']['instruction']

        if len(instruction) != 0: print(f"{datapoint} : {instruction['label']['instruction']}")

def evaluate_tag():
    """
        Evaluate the accuracy of the ATLP by returning the confusion matrix
    """

    pass


if __name__ == "__main__":
    print("This is ATLP module!")
