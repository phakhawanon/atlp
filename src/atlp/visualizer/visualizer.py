import matplotlib.pyplot as plt
import pinocchio as pin
import numpy as np
from pathlib import Path
from pinocchio.visualize import MeshcatVisualizer
import time

from ..modeller.motion.motion import (
    joint_lim_dict,        
    mjcf_path,
    _build_q,
)
# import meshcat

# file_path = 'data.json'
# report_file_path = 'report.txt'

_SIMULATION_STEP_SIZE = 50

def plot_joint_states(time_array, joint_arrays, subfields, quantity):
    """
        Plot joint states

        Args:
            time_array:
            joint_arrays:
            subfields:
            quantity:
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

def simulate_joint_arrays(all_time_array, all_joint_arrays, time_step: int = _SIMULATION_STEP_SIZE):
    """
        Open the browser and 3D simulate the given robot pose
    """
    model, _, collision_model, visual_model = pin.buildModelsFromMJCF(mjcf_path)
    model.createData()

    # viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
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