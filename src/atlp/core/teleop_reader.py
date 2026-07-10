import numpy as np
from numpy.typing import NDArray
import linecache
import re
import json
import cv2

from .interface import (
    get_root_directory,
)

_VIDEO_TYPE_TO_FILENAME = {
    "head": "camera_front_head_rgb.mp4",
    "left_wrist": "camera_left_wrist.mp4",
    "right_wrist": "camera_right_wrist.mp4",
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

    fields = [
        'state_body_joint_position',
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
def _get_video_duration_from_path(report_file_path: str) -> float:
    """
        Return the duration of the video of the datapoint in the specified path

        Args:
            report_file_path: Path to the report file of a datapoint

        Returns:
            Video duration in second of the datapoint

        .. todo::
            - Fix path
    """
    return float(linecache.getline(report_file_path,2).split(' ')[2])

def get_video_duration(
    datapoint : str,
) -> float:
    """
        Return the duration of the video of the datapoint

        Args:
            datapoint: Name of the datapoint

        Returns:
            Video duration in second of the datapoint
    """
    root_directory = get_root_directory()
    report_file_path = str(root_directory / datapoint / "report.txt")
    return _get_video_duration_from_path(report_file_path)

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
    root_directory = get_root_directory()
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
    duration = _get_video_duration_from_path(report_file_path)
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

    j = 1

    for i in range(0, ideal_array_width):

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
    dict,
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
    root_directory = get_root_directory()

    if quantity == "position":

        # Load data.json and shared values once for all fields
        report_file_path = str(root_directory / datapoint / "report.txt")
        file_path = root_directory / datapoint / "data.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        min_timestamp, max_timestamp = analyze_timestamp(data)
        diff = max_timestamp - min_timestamp
        duration = _get_video_duration_from_path(report_file_path)

        def _get(field, qty="position"):
            return convert_to_np_array(data, short_name_to_field_name[field], min_timestamp, diff, duration, report_file_path, quantity=qty)

        time_array_odom, joint_arrays_odom, *_ = _get("odom")
        time_array_left_arm, joint_arrays_left_arm, *_ = _get("left_arm_joints")
        time_array_right_arm, joint_arrays_right_arm, *_ = _get("right_arm_joints")
        time_array_leg, joint_arrays_leg, *_ = _get("leg_joints")
        time_array_head, joint_arrays_head, *_ = _get("head_joints")
        time_array_left_gripper, joint_arrays_left_gripper, *_ = _get("left_gripper")
        time_array_right_gripper, joint_arrays_right_gripper, *_ = _get("right_gripper")
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
    root_directory = get_root_directory()

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