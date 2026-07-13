"""
    Visualzier subpackage collects functions that are useful for visualizing the datapoints.

    Included in ```import atlp``` is atlp.visualizer.plotter, which requires only matplotlib installed.

    For other visualizer subpackages, pinocchio is  required for 3D simulation playback.
"""

from .plotter import (
    plot_joint_states,
)