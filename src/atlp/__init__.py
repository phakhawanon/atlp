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

from .core import *
from .visualizer import *