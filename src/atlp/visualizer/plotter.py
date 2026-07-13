"""
    Plotter module is used to plot joint state graphs.

    This module is automatically included when using ```import atlp````

    This module required that matplotlib is installed.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..modeller.motion.motion import joint_lim_dict

def plot_joint_states(
    time_array: NDArray[np.float64],
    joint_arrays: NDArray[np.float64],
    subfields: list[str],
    quantity: str,
) -> None:
    """
        Plot joint states

        Save the resulting figures as .jpg in the current directory.

        Args:
            time_array: The numpy array of timestamps obtained from teleop_reader.get_joint_states() 
            joint_arrays: The numpy array of joint states obtained from teleop_reader.get_joint_states()
            subfields: List of subfield names obtained from teleop_reader.get_joint_states()
            quantity: String obtained from teleop_reader.get_joint_states()
        
        .. todo::
            Configure saving path
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