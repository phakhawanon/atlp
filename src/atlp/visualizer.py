from .motion import (
    joint_lim_dict,        
)


import matplotlib.pyplot as plt
import pinocchio as pin
import numpy as np
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import time

# file_path = 'data.json'
# report_file_path = 'report.txt'

mjcf_path = Path("~/auto-task-labelling-pipeline/src/atlp/galbot_one_golf_collision_only.xml").expanduser()
_SIMULATION_STEP_SIZE = 50

def plot_joint_states(time_array, joint_arrays, subfields, quantity):
    """
        Plot joint states
    """    
    fig, axes = plt.subplots(len(subfields), 1, sharex=True, figsize=(5, len(subfields)*2.2))
    duration = time_array[-1] - time_array[0]
    fig.suptitle("Quantity: " + quantity)
   
    if len(subfields) == 1:
        enumerator = list()
        enumerator.append(axes)
        if len(joint_arrays.shape) == 2: joint_arrays = joint_arrays[0]
    else:
        enumerator = axes.flat
    for i, ax in enumerate(enumerator):
        if len(subfields) == 1: ax.plot(time_array,joint_arrays, color="blue")
        else: ax.plot(time_array, joint_arrays[i], color="blue")
        ax.set_xlim(0.0, duration)
        
        if subfields[i] in joint_lim_dict and quantity == "position":
            
            lim_lower, lim_upper = joint_lim_dict[subfields[i]]
            ax.set_ylim(lim_lower*1.2, lim_upper*1.2)
            ax.axhline(y=lim_lower, linestyle=":", color="red")
            ax.axhline(y=lim_upper, linestyle=":", color="red")

        # ax.plot(time_array,joint_arrays[i], color="blue")
        ax.set_title(subfields[i])
        ax.grid()
    
    plt.tight_layout()
    fig.savefig("test_joint_plot.jpg")

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

def _get_idx(model, joint_name):
    """
        Return index of the joint name from the given pinocchio model
    """
    return model.joints[model.getJointId(joint_name)].idx_q

def _build_q(model, all_joint_arrays, t):
    """
        Populate the model with the all_joint_arrays at time t
    """

    q = pin.neutral(model)
    
    q[0:7] = all_joint_arrays[0:7, t]
    
    for i, name in enumerate([f'leg_joint{j}' for j in range(1,5)]):
        q[_get_idx(model, name)] = all_joint_arrays[21+i, t]
    
    for i, name in enumerate([f'head_joint{j}' for j in range(1,3)]):
        q[_get_idx(model, name)] = all_joint_arrays[25+i, t]
    
    for i, name in enumerate([f'left_arm_joint{j}' for j in range(1,8)]):
        q[_get_idx(model, name)] = all_joint_arrays[7+i, t]
    
    for i, name in enumerate([f'right_arm_joint{j}' for j in range(1,8)]):
        q[_get_idx(model, name)] = all_joint_arrays[14+i, t]

    q[_get_idx(model, "left_gripper_joint")] = all_joint_arrays[27, t]
    q[_get_idx(model, "right_gripper_joint")] = all_joint_arrays[28, t]

    return q

def simulate_joint_arrays(all_time_array, all_joint_arrays, time_step: int = _SIMULATION_STEP_SIZE):
    """
        Open the browser and 3D simulate the given robot pose
    """
    model, _, collision_model, visual_model = pin.buildModelsFromMJCF(mjcf_path)
    data = model.createData()

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)        # opens browser tab automatically
    viz.loadViewerModel()
    print(viz.viewer.url())

    n = len(all_time_array)
    number_of_interations = (n + time_step - 1) / time_step
    delay_time = (all_time_array[-1] - all_time_array[0]) / number_of_interations

    for t in range(0, n, time_step):

        q = _build_q(model, all_joint_arrays, t)
        viz.display(q)
        time.sleep(delay_time)

    
if __name__ == "__main__":
    # main()
    print("Visualizer module is loaded successfully.")
