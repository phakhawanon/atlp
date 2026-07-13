"""
    Automatic Task Labelling Pipeline (ATLP) Package

    This python package is designed to act as a quality pipeline for galbot one golf foxtrot's teleoperation data.

    Provided with this package are python functions which can be used to
        - manage the teleoperation dataset. (atlp.core)
        - automatically label the teleoperation dataset with English instruction. (atlp.modeller.label)
        - automatically detect whether the given teleoperation data contains self-collision or not. (atlp.modeller.motion)
        - visualize the teleoperation dataset (atlp.visualizer)
        - evaluate the quality pipeline (atlp.evaluator)

    This python package also supports manual labelling for the teleoperation dataset, 
    though it is better to refer to ATLP GUI package for a more convenience GUI implementation of this package.

    The package is also written so that it is easy to be extended. Namely, if you want to modify this package to
        - accept data from any robot or platform, modify atlp.core.teleop_reader
        - replace the current data storage that uses only header.json, modify atlp.core.interface
        - add more automatic labelling pipelines (e.g. motion smoothness, or brightness), modify atlp.modeller
        - add more visualization tool, modify atlp.visualizer
        - add more means to evaluate the automatic labelling pipelines, modify atlp.evaluator

    As the name suggested, all subpackages reference atlp.core. 
    Therefore, correctly modifying the atlp.core do not break other subpackages.
    
    For the terms used in this package and repository, please refer to the README.md of this directory.

    Note:
        - Importing atlp only imports atlp.core and atlp.visualizer.plotter.
            If you want to use other subpackages, import them explicitly.
            This is the case because other subpackages are more heavy and application-specific.

    Example:
        >>> import atlp # This import atlp.core and atlp.visualizer.plotter
        >>> import atlp.modeller.label
        >>> atlp.set_root(root_to_dataset)
        >>> atlp.populate_header() # Track the data in the dataset
        >>> atlp.list_datapoints() # Print all details of all datapoints
        >>> atlp.show_statistics() # See how many unlabelled datapoints are there in the dataset
        >>> atlp.modeller.label.label() # Label all unlabelled datapoints
        >>> atlp.display_instruction() # Print all instructions of all datapoints

        .. todo::
        - Implement trivial error handling (input type mismatch, etc) for all functions

    Note:
        TBD Note

        Datapoint is a folder inside the dataset that ends with "_record0"

        The datapoint folder should at least contain camera_front_head_rgb.mp4, which is the RGB video output from the Galbot G1 head camera
"""

import os
from pathlib import Path


def _default_robot_data_dir() -> Path:
    """Locate robot_data/ (URDF/MJCF/meshes), which ships as a sibling of src/, not inside the atlp package."""
    env = os.environ.get("ATLP_ROBOT_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "robot_data"


ROBOT_DATA_DIR = _default_robot_data_dir()

from .core import *
from .visualizer import *