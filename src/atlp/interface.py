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
import numpy as np
import linecache
import re
import cv2
from numpy.typing import NDArray

# Leave this empty if using in this directory
# must end with /
root_directory = Path(".").expanduser()

# The tuple of endings that will be treated as a datapoint
datapoint_ending = ("_record0")

# Set to true to enable collecting actual fields
enable_actual_field = True

# The list of default tags
default_tag_list = ["Pouring", "Failed"]

header_path = root_directory / "header.json"
label_field_values = {
    "instruction": [str],
    "tags": [list],
    "is_failed": [None, False, True],
}

vision_field_values = {
    "brightness": [int],
}

motion_field_values = {
    "is_self_collide": [None, False, True],
}

field_to_report_name_dict = {
    "state_body_joint_position"         :   "/body/joint_states",
    "state_front_head_joint"            :   "/head/joint_states",
    "state_left_arm_joint_position"     :   "/left_arm/joint_states",
    "state_right_arm_joint_position"    :   "/right_arm/joint_states",
    "state_left_arm_gripper_width"      :   "/left_arm_gripper/joint_states",
    "state_right_arm_gripper_width"     :   "/right_arm_gripper/joint_states",
    "odom"                              :   "/odom",            
    # "leg_joints"                        :   "/body/joint_states",
    # "head_joints"                       :   "/head/joint_states",
    # "left_arm_joints"                   :   "/left_arm/joint_states",
    # "right_arm_joints"                  :   "/right_arm/joint_states",
    # "left_gripper"                      :   "/left_arm_gripper/joint_states",
    # "right_gripper"                     :   "/right_arm_gripper/joint_states",
}

short_name_to_field_name = {        
    "state_body_joint_position"         :   "state_body_joint_position",
    "state_front_head_joint"            :   "state_front_head_joint",
    "state_left_arm_joint_position"     :   "state_left_arm_joint_position",
    "state_right_arm_joint_position"    :   "state_right_arm_joint_position",
    "state_left_arm_gripper_width"      :   "state_left_arm_gripper_width",
    "state_right_arm_gripper_width"     :   "state_right_arm_gripper_width",
    "odom"                              :   "odom",            
    "leg_joints"                        :   "state_body_joint_position",
    "head_joints"                       :   "state_front_head_joint",
    "left_arm_joints"                   :   "state_left_arm_joint_position",
    "right_arm_joints"                  :   "state_right_arm_joint_position",
    "left_gripper"                      :   "state_left_arm_gripper_width",
    "right_gripper"                     :   "state_right_arm_gripper_width",
    "base"                              :   "odom",
    "odometry"                          :   "odom",
}

def get_root_directory() -> Path:
    """
        Return the current root directory as Path object

        Returns:
            Path object of the root directory
    """
    return root_directory

# Find min and max timestamps from the data
def analyze_timestamp(data: dict) -> tuple[int, int]:
    """
        Find the min and max timestamps from the data

        Args:
            data: Dictionary of the data.json in each datapoint.

        Returns:
            (min, max)
                min/max are the minumum and maximum timestamps in int
    """
    
    # Initialize min and max timestamps to the first element of odom
    min_timestamp = data['data']['odom'][0]['timestamp']
    max_timestamp = min_timestamp

    fields = ['state_body_joint_position',
              'state_front_head_joint',
              'state_left_arm_joint_position',
              'state_right_arm_joint_position',
              'state_left_arm_gripper_width',
              'state_right_arm_gripper_width',
              'odom'
              ]

    for field in fields:
        for item in data['data'][field]:
            timestamp_value = item['timestamp']
            if timestamp_value < min_timestamp:
                min_timestamp = timestamp_value
            if timestamp_value > max_timestamp:
                max_timestamp = timestamp_value
    
    return min_timestamp, max_timestamp

# Return duration of video in s
# TODO: Fix path
def get_video_duration(report_file_path: str) -> float:
    """
        Return the duration of the video of the datapoint in the specified path

        Args:
            report_file_path: Path to the report file of a datapoint

        Returns:
            Video duration in second of the datapoint

        ..todo::
            - Fix path
    """
    return float(linecache.getline(report_file_path,2).split(' ')[2])

def get_field_message_count(field: str, report_file_path: str) -> int:
    """
        Return the message count of a field in the span of the datapoint.

        Args:
            field: The name of the field in the keys of field_to_report_name_dict
            report_file_path: The path to the report.text of the datapoint

        Returns:
            The number of message count of the field as specified in the report.txt of the datapoint
            
    """

    field_report_name = field_to_report_name_dict[field]

    with open(report_file_path) as f:
        for i, line in enumerate(f, start=1):
            line_list = re.split(r'\s+', line.strip())
            if len(line_list) >= 2:
                if line_list[1] == field_report_name:
                    return int(line_list[3])

# diff = max_timestamp - min_timestamp
# return time_array, joint_arrays, and subfields (subfield names)
def convert_to_np_array(
    data : dict,
    field : str,
    min_timestamp : int,
    diff : float,
    duration : float,
    report_file_path : str,
    quantity : str = "position",
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """
        Retrieve the content of the data.json for the specific datapoint specified by the path to its report.txt

        Args:
            data: Dictionary of data.json of the datapoint
            field: String of field name that is the key of field_to_report_name_dict
            min_timestamp: The minimum timestamp of the datapoint
            diff: Difference between the maximum and minimum timestamps of the datapoint
            duration: Duration of the datapoint in second
            report_file_path: The path to the report.txt of the datapoint
            quantity: Choose from "position", "velocity", and "effort"

        Returns:
            (time_array, joint_arrays, subfields_list)
                - time_array is (1, N) numpy array that contains the time in second of each timestamp
                - joint_arrays is (M, N) numpy array that contains the joint quantities of each timestamp
                - subfields_list is a list of strings that describes the quantity of each row of joint_arrays
                where M is the length of subfields_list

                and N is the number of timestamps of the datapoint
    """
    
    if field == "state_left_arm_gripper_width": subfields = ['left_gripper']
    elif field == "state_right_arm_gripper_width": subfields = ['right_gripper']
    elif field == "odom":
        if quantity == "position": subfields = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        elif quantity == "velocity": subfields = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        else: raise ValueError(f"For field odom, quantity {quantity} is invalid.")
    else: subfields = data['data'][field][0]['names']    

    subfield_count = len(subfields)
    
    field_message_count = get_field_message_count(field, report_file_path)

    # Initialize time array
    time_array = np.ndarray((field_message_count), float)

    # Initialize joint arrays
    joint_arrays = np.ndarray((subfield_count, field_message_count), float)
    # for i in range(0, subfield_count):
        # joint_arrays.append(np.ndarray((field_message_count), float))
    
    # Populate time array and joint arrays
    i = 0

    if field == "odom":

        if quantity == "position": odom_field = "pose"
        elif quantity == "velocity": odom_field = "twist"
            
        for item in data['data'][field]:

            timestamp_value = item['timestamp']
            time_value = (timestamp_value - min_timestamp) * duration / diff
            time_array[i] = time_value

            if quantity == "position":

                joint_arrays[0, i] = item['data'][odom_field]["position"]["x"]
                joint_arrays[1, i] = item['data'][odom_field]["position"]["y"]
                joint_arrays[2, i] = item['data'][odom_field]["position"]["z"]
                joint_arrays[3, i] = item['data'][odom_field]["orientation"]["x"]
                joint_arrays[4, i] = item['data'][odom_field]["orientation"]["y"]
                joint_arrays[5, i] = item['data'][odom_field]["orientation"]["z"]
                joint_arrays[6, i] = item['data'][odom_field]["orientation"]["w"]

            else:
                
                joint_arrays[0, i] = item['data'][odom_field]["linear"]["x"]
                joint_arrays[1, i] = item['data'][odom_field]["linear"]["y"]
                joint_arrays[2, i] = item['data'][odom_field]["linear"]["z"]
                joint_arrays[3, i] = item['data'][odom_field]["angular"]["x"]
                joint_arrays[4, i] = item['data'][odom_field]["angular"]["y"]
                joint_arrays[5, i] = item['data'][odom_field]["angular"]["z"]

            i += 1

    else:
        
        for item in data['data'][field]:
            
            timestamp_value = item['timestamp']
            time_value = (timestamp_value - min_timestamp) * duration / diff 
            time_array[i] = time_value
            joint_values = item[quantity]

            for j in range(0, subfield_count):
                joint_arrays[j, i] = joint_values[j]
                #print(joint_values[j])
        
            i += 1

    return time_array, joint_arrays, subfields, quantity

def get_joint_states(
    datapoint: str,
    field: str,
    quantity: str = "position",
    frequency: int = None,
    total_time: float = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str], str]:
    """
        Get the joint states of the datapoint as numpy arrays

        Assume that the given datapoint exists

        Args:
            datapoint: Name of the datapoint
            field: Name of the field as specified by the keys of short_name_to_field_name dict
            quantity: Choose from "position", "velocity", or "effort"
            frequency: If provided along with total_time, resample the output to this frequency (Hz)
            total_time: Duration in seconds to resample over; required when frequency is set

        Returns:
            (time_array, joint_arrays, subfields, quantity)
                - time_array: (N,) timestamps in seconds
                - joint_arrays: (M, N) joint quantities
                - subfields: list of subfield name strings
                - quantity: the quantity string used
    """
    report_file_path = str(root_directory / datapoint / "report.txt")
    file_path = root_directory / datapoint / "data.json"

    # Handle aliases and invalid names
    if field in short_name_to_field_name: field = short_name_to_field_name[field]

    else:

        print(f"Invalid field name: Field {field} does not exist.")
        return

    if quantity not in ['position', 'velocity', 'effort']:

        print(f"Invalid quantity. Quantity {quantity} does not exist.")
        return

    if field == "odom" and quantity not in ['position', 'velocity']:

        print(f"Invalid quantity for field odom. Quantity {quantity} does not exist.")
        return

    if field in ['state_left_arm_gripper_width', 'state_right_arm_gripper_width'] and quantity != "position":

        print(f"Invalid quantity for field {field}. Quantity {quantity} does not exist.")
        return

    with open(file_path,'r') as f:
        data = json.load(f)
    
    # Get min and max timestamps of the dataset
    min_timestamp, max_timestamp = analyze_timestamp(data)
    # print((min_timestamp, max_timestamp))

    # Calculate their difference
    diff = max_timestamp - min_timestamp
    # print(diff)

    # Get duration in s
    duration = get_video_duration(report_file_path)
    # print(duration)

    time_array, joint_arrays, subfields, qty = convert_to_np_array(
        data, field, min_timestamp, diff, duration, report_file_path, quantity=quantity
    )

    if frequency is not None and total_time is not None:
        num_samples = int(frequency * total_time)
        new_time_array = np.linspace(time_array[0], time_array[0] + total_time, num_samples)
        joint_arrays = linear_interpolate_to_frequency(time_array, joint_arrays, frequency, total_time)
        time_array = new_time_array

    return time_array, joint_arrays, subfields, qty

def _linear_interpolation(
    ideal_time_array : NDArray[np.float64],
    time_array : NDArray[np.float64],
    joint_arrays : NDArray[np.float64],
) -> NDArray[np.float64]:
    """
        Linear interpolate the joint_arrays to all timestamps in ideal_time_array

        Assume that ideal_time_array is longer than time_array
    
        Args:
            ideal_time_array: (1, N) Timestamps of the wanted interpolated result
            time_array: (1, n) Timestamps of the current joint_arrays
            joint_arrays: (M, n) Joint arrays of the joint quantities

        Returns:
            Interpolated joint arrays with size (M, N)

        where
            M is the #rows of joint_arrays

            N is the #cols of ideal_time_array

            n is the #cols of joint_arrays and time_array
        

        .. todo::
            - Handle low j
    """
    array_width = time_array.shape[0]
    ideal_array_width = ideal_time_array.shape[0]
    interpolated_size = (joint_arrays.shape[0], ideal_array_width)
    interpolated_joint_arrays = np.ndarray(interpolated_size, float)

    if ideal_array_width < array_width:
        indices = [i * array_width // ideal_array_width for i in range(ideal_array_width)]
        return joint_arrays[:, indices]

    for i in range(0, ideal_array_width):

        j = 1

        try:
            
            while time_array[j] < ideal_time_array[i] and j < array_width-1: j += 1

        finally:
        
            x0 = time_array[j-1]
            x1 = time_array[j]
        
            y0 = joint_arrays[:, j-1]
            y1 = joint_arrays[:, j]

            m = (y1 - y0) / (x1 - x0)

        x = ideal_time_array[i]
        y = y0 + m * (x - x0) # Element-wise multiplication                

        interpolated_joint_arrays[:, i] = y

    return interpolated_joint_arrays

def linear_interpolate_to_frequency(
    time_array : NDArray[np.float64],
    joint_arrays : NDArray[np.float64],
    frequency : int,
    total_time : float,
) -> NDArray[np.float64]:
    """
    Interpolate joint_arrays to a fixed sampling frequency over a given duration.

    Builds an ideal_time_array from frequency and total_time, then delegates to
    _linear_interpolation().

    Args:
        time_array: (n,) Timestamps of the current joint_arrays
        joint_arrays: (M, n) Joint arrays of the joint quantities
        frequency: Desired output frequency in Hz (must be > 0)
        total_time: Duration of the output in seconds (must be > 0)

    Returns:
        Interpolated joint arrays with size (M, N) where N = frequency * total_time
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    if total_time <= 0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if time_array.shape[0] == 0:
        raise ValueError("time_array must not be empty")
    if joint_arrays.shape[-1] != time_array.shape[0]:
        raise ValueError(
            f"joint_arrays last dimension ({joint_arrays.shape[-1]}) must match "
            f"time_array length ({time_array.shape[0]})"
        )

    num_samples = int(frequency * total_time)
    if num_samples == 0:
        raise ValueError(
            f"frequency * total_time = {frequency * total_time} produces 0 samples"
        )

    ideal_time_array = np.linspace(time_array[0], time_array[0] + total_time, num_samples)

    return _linear_interpolation(ideal_time_array, time_array, joint_arrays)

def get_all_joint_states(
    datapoint : str,
    quantity : str = "position",
    ideal_time_array : list[float] = list(),
    frequency : int = None,
    total_time : float = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    dict,
    dict
]:
    """
        Get all joint states as one large numpy arrays

        Currently only suppose quantity = "position"

        Args:
            datapoint: Name of the datapoint
            quantity: Must be "position"
            ideal_time_array: (N,) If specified, the joint states will be interpolated to these timestamps.
            frequency: If provided along with total_time, build ideal_time_array at this frequency (Hz).
                       Takes priority over the ideal_time_array parameter.
            total_time: Duration in seconds; required when frequency is set.

        Returns:
            (ideal_time_array, joint_arrays, field_slicers_dict, subfields_dict)
                - ideal_time_array: (N,)
                - joint_arrays: (29, N)
                - field_slicers_dict: Contain slice object describing the rows of joint_arrays
                - subfields_dict: Contain subfields of all fields in joint_arrays

        where N = frequency * total_time if frequency/total_time are given,
        or the length of ideal_time_array if provided,
        or the length of the field with highest frequency otherwise.
    """
    if quantity == "position":

        time_array_odom, joint_arrays_odom, *_ = get_joint_states(datapoint, "odom", quantity="position")
        time_array_left_arm, joint_arrays_left_arm, *_ = get_joint_states(datapoint, "left_arm_joints", quantity="position")
        time_array_right_arm, joint_arrays_right_arm, *_ = get_joint_states(datapoint, "right_arm_joints", quantity="position")
        time_array_leg, joint_arrays_leg, *_ = get_joint_states(datapoint, "leg_joints", quantity="position")
        time_array_head, joint_arrays_head, *_ = get_joint_states(datapoint, "head_joints", quantity="position")
        time_array_left_gripper, joint_arrays_left_gripper, *_ = get_joint_states(datapoint, "left_gripper", quantity="position")
        time_array_right_gripper, joint_arrays_right_gripper, *_ = get_joint_states(datapoint, "right_gripper", quantity="position")
        time_array_list = [
            time_array_odom,
            time_array_left_arm,
            time_array_right_arm,
            time_array_leg,
            time_array_head,
            time_array_left_gripper,
            time_array_right_gripper,
        ]

        # Finding ideal_time_array
        if frequency is not None and total_time is not None:
            num_samples = int(frequency * total_time)
            ideal_time_array = np.linspace(time_array_odom[0], time_array_odom[0] + total_time, num_samples)

        elif len(ideal_time_array) == 0:

            max_time_array_length = len(time_array_odom)
            ideal_time_array = time_array_odom

            for time_array in time_array_list:

                if len(time_array) > max_time_array_length:

                    max_time_array_length = len(time_array)
                    ideal_time_array = time_array

        ideal_time_array_length = len(ideal_time_array)
        joint_arrays = np.ndarray((29, ideal_time_array_length), float)

        # Populate joint_arrays
        joint_arrays[0:7, :] = _linear_interpolation(ideal_time_array, time_array_odom, joint_arrays_odom)
        joint_arrays[7:14, :] = _linear_interpolation(ideal_time_array, time_array_left_arm, joint_arrays_left_arm)
        joint_arrays[14:21, :] = _linear_interpolation(ideal_time_array, time_array_right_arm, joint_arrays_right_arm)
        joint_arrays[21:25, :] = _linear_interpolation(ideal_time_array, time_array_leg, joint_arrays_leg)
        joint_arrays[25:27, :] = _linear_interpolation(ideal_time_array, time_array_head, joint_arrays_head)
        joint_arrays[27, :] = _linear_interpolation(ideal_time_array, time_array_left_gripper, joint_arrays_left_gripper)
        joint_arrays[28, :] = _linear_interpolation(ideal_time_array, time_array_right_gripper, joint_arrays_right_gripper)

        field_slicers_dict = {
            "odom": slice(0,7),
            "left_arm_joints": slice(7, 14),
            "right_arm_joints": slice(14, 21),
            "leg_joints": slice(21, 25),
            "head_joints": slice(25, 27),
            "left_gripper": 27,
            "right_gripper": 28,
        }

        subfields_dict = {            
            "odom": ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'],
            "left_arm_joints": [
                'left_arm_joint1',
                'left_arm_joint2',
                'left_arm_joint3',
                'left_arm_joint4',
                'left_arm_joint5',
                'left_arm_joint6',
                'left_arm_joint7',
            ],
            "right_arm_joints": [
                'right_arm_joint1',
                'right_arm_joint2',
                'right_arm_joint3',
                'right_arm_joint4',
                'right_arm_joint5',
                'right_arm_joint6',
                'right_arm_joint7',
             ],
            "leg_joints": ['leg_joint1', 'leg_joint2', 'leg_joint3', 'leg_joint4'],
            "head_joints": ['head_joint1', 'head_joint2'],
            "left_gripper": ['left_gripper'],
            "right_gripper": ['right_gripper'],
        }

        return ideal_time_array, joint_arrays, field_slicers_dict, subfields_dict

    else:

        print(f"Invalid quantity. Quantity {quantity} does not exist.")
        return

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

        return None, value

    default_value = field_values[field][0]

    if isinstance(default_value, type):
        try:
            if default_value is list:

                if isinstance(value, str):
                    # print(value)
                    value = json.loads(value)
                    # print(value)

                # print(f"I am not a string {value}")

            else:
                value = default_value(value)

            # print(isinstance(value, default_value))
            # print(default_value)
            # print(type(value))
            return isinstance(value, default_value), value
        except Exception:
            return False, value

    if value in ["false", "False"]: value = False
    if value in ["true", "True"]: value = True
    if value in ["none", "None"]: value = None

    if value in field_values[field]: return True, value
    else: return False, value

def set_enable_actual_field(_enable_actual_field: bool) -> None:
    """
        Change enable_actual_field
    """
    global enable_actual_field
    if isinstance(_enable_actual_field, bool): enable_actual_field = _enable_actual_field

def _is_modelled(
    datapoint: str,
    use_dict: dict = dict(),
) -> bool:
    """
        Determine whether the datapoint has been modelled.
    """
    datapoint_tags = get_single_datapoint(datapoint,
                                          main_field="label",
                                          field="tags",
                                          use_dict=use_dict,
                                          )
    if len(datapoint_tags) == 0: return False
    return True

def _is_checked(
    datapoint: str,
    use_dict: dict = dict(),
) -> bool:
    """
        Determine whether the datapoint has been manually checked
    """
    datapoint_tags = get_single_datapoint(datapoint,
                                          main_field="label",
                                          field="tags",
                                          use_dict=use_dict,
                                          from_actual=True,
                                          )
    if len(datapoint_tags) == 0: return False
    return True

def get_single_datapoint(
    datapoint: str,
    main_field: str,
    field: str,
    from_actual: bool = False,
    use_dict: dict = dict(),
):
    if main_field not in ["label", "vision", "motion"]:
        raise ValueError(f"Invalid main field. There is no {main_field}.")

    is_invalid_field = False

    if main_field == "label" and field not in label_field_values: is_invalid_field = True
    elif main_field == "vision" and field not in vision_field_values: is_invalid_field = True
    elif main_field == "motion" and field not in motion_field_values: is_invalid_field = True

    if is_invalid_field:
        raise ValueError(f"Invalid field. There is no {field} in {main_field}")

    labels = []
    visions = []
    motions = []

    if main_field == "label": labels = [field]
    elif main_field == "vision": visions = [field]
    elif main_field == "motion": motions = [field]

    data_from_get_datapoint = get_datapoint(datapoint,
                                            from_actual=from_actual,
                                            use_dict=use_dict,
                                            labels=labels,
                                            visions=visions,
                                            motions=motions,    
                                            )
    wanted_data = data_from_get_datapoint[main_field][field]
    return wanted_data

def filter(
  use_dict: dict = dict(),
  is_checked = None,
  is_modelled = None,
  # is_inconsistent = None,
  model_tag = None,
  actual_tag = None,
  model_is_failed = None,
  actual_is_failed = None,  
):
    if len(use_dict)==0: data=load_data()
    else: data=use_dict
        
    datapoint_set = set(data["datapoints"].keys())
    to_be_deleted_datapoint_set = set() 
    
    tag_list = tag_get(use_dict=data, from_actual=False)
    actual_tag_list = tag_get(use_dict=data, from_actual=True)

    filter_model_tag = True
    filter_actual_tag = True
    
    try:
        if str(model_tag) not in tag_list: filter_model_tag = False
    except Exception: filter_model_tag = False
                
    try:
        if str(actual_tag) not in actual_tag_list: filter_actual_tag = False
    except Exception: filter_actual_tag = False    

    if filter_model_tag:
        model_tag = str(model_tag)
        for datapoint in datapoint_set:
            datapoint_tags = get_single_datapoint(datapoint,
                                                  main_field="label",
                                                  field="tags",
                                                  use_dict=data,
                                                  )
            if model_tag not in datapoint_tags:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()
    
    if filter_actual_tag:
        actual_tag = str(actual_tag)
        for datapoint in datapoint_set:
            datapoint_tags = get_single_datapoint(datapoint,
                                                  main_field="label",
                                                  field="tags",
                                                  use_dict=data,
                                                  from_actual=True,
                                                  )
            if actual_tag not in datapoint_tags:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if model_is_failed in [True, False]:
        for datapoint in datapoint_set:
            datapoint_is_failed = get_single_datapoint(datapoint,
                                                       main_field="label",
                                                       field="is_failed",
                                                       use_dict=data,
                                                       )
            if datapoint_is_failed != model_is_failed:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()
    
    if actual_is_failed in [True, False]:
        for datapoint in datapoint_set:
            datapoint_is_failed = get_single_datapoint(datapoint,
                                                       main_field="label",
                                                       field="is_failed",
                                                       use_dict=data,
                                                       from_actual=True,
                                                       )
            if datapoint_is_failed != actual_is_failed:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if is_modelled in [True, False]:
        for datapoint in datapoint_set:
            if is_modelled != _is_modelled(datapoint, use_dict=data):
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if is_checked in [True, False]:
        for datapoint in datapoint_set:
            if is_checked != _is_checked(datapoint, use_dict=data):
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    datapoint_list = list(datapoint_set)
    return datapoint_list       
     

def get_datapoint(
    datapoint: str,
    from_actual: bool = False,
    use_dict: dict = dict(),
    labels: list = [],
    visions: list = [],
    motions: list = [],
    get_all: bool = False,
):
    """
        Safely get the data from the specified datapoint
    """
    return_dict = dict()
    return_labels = dict()
    return_visions = dict()
    return_motions = dict()
    
    # Load data
    if len(use_dict) == 0:

        data = load_data()

    else:

        data = use_dict.copy()

    # Handle invalid datapoint name
    if datapoint not in data["datapoints"]:

        print(f"Cannot find datapoint: {datapoint} does not exist.")
        
        return dict()

    # Check if there is "actual"
    if from_actual and "actual" not in data["datapoints"][datapoint]:

        print(f"Datapoint {datapoint} does not have actual fields")

        return dict()

    elif not from_actual: data_to_retrieve = data["datapoints"][datapoint]["model"]
    else: data_to_retrieve = data["datapoints"][datapoint]["actual"]

    if get_all:
        labels = label_field_values.keys()
        visions = vision_field_values.keys()
        motions = motion_field_values.keys()

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
        is_valid_value, value = _is_valid_field_value(value, label, label_field_values)

        if is_valid_value is None: print(f"Invalid label field name: Field '{label}' does not exist.")
        elif is_valid_value: data_to_modify["label"][label] = value
        else: print(f"Invalid label field value: Field '{label}' does not have value '{labels[label]}'")

    for vision in visions:
        
        value = visions[vision]
        is_valid_value, value = _is_valid_field_value(value, vision, vision_field_values)

        if is_valid_value is None: print(f"Invalid vision field name: Field '{vision}' does not exist.")
        elif is_valid_value: data_to_modify["vision"][vision] = value
        else: print(f"Invalid vision field value: Field '{vision}' does not have value '{visions[vision]}'")

    for motion in motions:
                
        value = motions[motion]
        is_valid_value, value = _is_valid_field_value(value, motion, motion_field_values)

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
    # tag_list = data["tag_list"]
    tag_list = default_tag_list
    actual_tag_list = []

    for datapoint in data["datapoints"]:

        count_datapoint += 1

        # if data["datapoints"][datapoint]["is_labelled"]: count_labelled += 1
        # if data["datapoints"][datapoint]["is_failed"]: count_failed += 1
        # if data["datapoints"][datapoint]["is_checked"]: count_checked += 1

        tag_list += data["datapoints"][datapoint]["model"]["label"]["tags"]
        tag_list = list(set(tag_list))

        if enable_actual_field:
            
            actual_tag_list += data["datapoints"][datapoint]["actual"]["label"]["tags"]
            actual_tag_list = list(set(actual_tag_list))

    data["count_datapoint"] = count_datapoint
    # data["count_labelled"] = count_labelled
    # data["count_failed"] = count_failed
    # data["count_checked"] = count_checked
    data["tag_list"] = tag_list
    data["actual_tag_list"] = actual_tag_list
    
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

        load_data()

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
        "actual_tag_list": [],
        "datapoints": {},
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

    update_statistics()
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


def tag_get(
    use_dict: dict = dict(),
    from_actual: bool = False,
) -> list:
    """
        Return tag_list of the header file

        Returns:
            List of all tags inside the header.json
    """
    if len(use_dict) == 0: data = load_data()
    else: data = use_dict 

    if from_actual: return data["actual_tag_list"]
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
    print(f"actual_tag_list: {data['actual_tag_list']}")
    
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
    return list(data["datapoints"].keys())

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

_VIDEO_TYPE_TO_FILENAME = {
    "head": "camera_front_head_rgb.mp4",
    "left_wrist": "camera_left_wrist.mp4",
    "right_wrist": "camera_right_wrist.mp4",
}

def get_video_frames(
    datapoint: str,
    frame_ids: int | list[int],
    video_type: str,
) -> NDArray[np.uint8]:
    """
    Get specific frames from a video in the datapoint as RGB numpy arrays.

    Args:
        datapoint: Name of the datapoint folder (e.g. "20260213_140653_record0")
        frame_ids: A single frame index or a list of frame indices to retrieve
        video_type: Camera type — one of "head", "left_wrist", or "right_wrist"

    Returns:
        RGB numpy array of shape (N, H, W, 3) where N = number of requested frames
    """
    if video_type not in _VIDEO_TYPE_TO_FILENAME:
        raise ValueError(
            f"Invalid video_type '{video_type}'. Must be one of {list(_VIDEO_TYPE_TO_FILENAME.keys())}"
        )

    if isinstance(frame_ids, int):
        frame_ids = [frame_ids]

    video_path = str(root_directory / datapoint / _VIDEO_TYPE_TO_FILENAME[video_type])
    cap = cv2.VideoCapture(video_path)

    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {fid} from {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.stack(frames)


if __name__ == "__main__":
    print("This is ATLP module!")
