import json
import numpy as np
import matplotlib.pyplot as plt
import linecache
import re

from . import atlp

# file_path = 'data.json'
# report_file_path = 'report.txt'

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

# left_gripper and right_gripper do not appear in data.json,
# but are made by convert_to_np_array for convenience 
joint_lim_dict = {
    "leg_joint1"        :   (0.0, 0.9374),
    "leg_joint2"        :   (0.0, 2.5847),
    "leg_joint3"        :   (0.0, 2.3262),
    "leg_joint4"        :   (-1.5906, 1.5906),
    "head_joint1"       :   (-1.5208, 1.5208),
    "head_joint2"       :   (-0.2143461, 0.4935988),
    "left_arm_joint1"   :   (-3.00432619, 3.00432619),
    "left_arm_joint2"   :   (-1.608062789, 1.608062789),
    "left_arm_joint3"   :   (-2.916972222, 2.916972222),
    "left_arm_joint4"   :   (-2.5679938779914944, 1.869862177),
    "left_arm_joint5"   :   (-2.916972222, 2.916972222),
    "left_arm_joint6"   :   (-0.8226646259971648, 0.7353981633974483),
    "left_arm_joint7"   :   (-1.538202778, 1.538202778),
    "right_arm_joint1"  :   (-3.00432619, 3.00432619),
    "right_arm_joint2"  :   (-1.608062789, 1.608062789),
    "right_arm_joint3"  :   (-2.916972222, 2.916972222),
    "right_arm_joint4"  :   (-1.869862177, 2.5679938779914944),
    "right_arm_joint5"  :   (-2.916972222, 2.916972222),
    "right_arm_joint6"  :   (-0.7353981633974483, 0.8226646259971648),
    "right_arm_joint7"  :   (-1.538202778, 1.538202778),
    "left_gripper"      :   (0.0, 1.703),
    "right_gripper"     :   (0.0, 1.703)
    }


# Find min and max timestamps from the data
def analyze_timestamp(data):

    
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
def get_video_duration(report_file_path):
    return float(linecache.getline(report_file_path,2).split(' ')[2])

def get_field_message_count(field, report_file_path):

    field_report_name = field_to_report_name_dict[field]

    with open(report_file_path) as f:
        for i, line in enumerate(f, start=1):
            line_list = re.split(r'\s+', line.strip())
            if len(line_list) >= 2:
                if line_list[1] == field_report_name:
                    return int(line_list[3])

# diff = max_timestamp - min_timestamp
# return time_array, joint_arrays, and subfields (subfield names)
def convert_to_np_array(data, field, min_timestamp, diff, duration, report_file_path):
    
    if field == "state_left_arm_gripper_width": subfields = ['left_gripper']
    elif field == "state_right_arm_gripper_width": subfields = ['right_gripper']
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
    for item in data['data'][field]:
        timestamp_value = item['timestamp']
        time_value = (timestamp_value - min_timestamp) * duration / diff 
        

        time_array[i] = time_value
        joint_values = item['position']

        for j in range(0, subfield_count):
            joint_arrays[j, i] = joint_values[j]
            #print(joint_values[j])
        
        i += 1

    return time_array, joint_arrays, subfields

# Obsolete
def _plot_field (field, min_timestamp, diff, duration):

    time_array, joint_arrays, subfields = convert_to_np_array(field, min_timestamp, diff, duration)
    fig, axes = plt.subplots(len(subfields), 1, sharex=True, figsize=(5, len(subfields)*2.2))
   
    if len(subfields) == 1:
        enumerator = list()
        enumerator.append(axes)
    else:
        enumerator = axes.flat
    for i, ax in enumerate(enumerator):
        ax.plot(time_array, joint_arrays[i])
        ax.set_xlim(0.0, duration)
        lim_lower, lim_upper = joint_lim_dict[subfields[i]]
        ax.set_ylim(lim_lower*1.2, lim_upper*1.2)
        ax.axhline(y=lim_lower, linestyle=":", color="red")
        ax.axhline(y=lim_upper, linestyle=":", color="red")
        ax.plot(time_array,joint_arrays[i], color="blue")
        ax.set_title(subfields[i])
        ax.grid()
    
    plt.tight_layout()
    fig.savefig("test_joint_plot.jpg")

def plot_joint_states(time_array, joint_arrays, subfields):
    """
        Plot joint states
    """    
    fig, axes = plt.subplots(len(subfields), 1, sharex=True, figsize=(5, len(subfields)*2.2))
    duration = time_array[-1] - time_array[0]
   
    if len(subfields) == 1:
        enumerator = list()
        enumerator.append(axes)
    else:
        enumerator = axes.flat
    for i, ax in enumerate(enumerator):
        ax.plot(time_array, joint_arrays[i])
        ax.set_xlim(0.0, duration)
        lim_lower, lim_upper = joint_lim_dict[subfields[i]]
        ax.set_ylim(lim_lower*1.2, lim_upper*1.2)
        ax.axhline(y=lim_lower, linestyle=":", color="red")
        ax.axhline(y=lim_upper, linestyle=":", color="red")
        ax.plot(time_array,joint_arrays[i], color="blue")
        ax.set_title(subfields[i])
        ax.grid()
    
    plt.tight_layout()
    fig.savefig("test_joint_plot.jpg")

def get_joint_states(datapoint: str, field: str):
    """
        Get the joint states of the datapoint as a list of np.arrays

        Assume that the given datapoint exists
    """
    report_file_path = str(atlp.root_directory / datapoint / "report.txt")
    file_path = atlp.root_directory / datapoint / "data.json"

    if field in short_name_to_field_name: field = short_name_to_field_name[field]

    else:

        print(f"Invalid field name: Field {field} does not exist.")
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

    return convert_to_np_array(data, field, min_timestamp, diff, duration, report_file_path)

# def main():
        
#     global data

#     with open(file_path,'r') as f:
#         data = json.load(f)
    
#     # Get min and max timestamps of the dataset
#     min_timestamp, max_timestamp = analyze_timestamp()
#     print((min_timestamp, max_timestamp))

#     # Calculate their difference
#     diff = max_timestamp - min_timestamp
#     print(diff)

#     # Get duration in s
#     duration = get_video_duration()
#     print(duration)

#     plot_field("state_right_arm_joint_position", min_timestamp, diff, duration)


if __name__ == "__main__":
    # main()
    print("Visualizer module is loaded successfully.")
