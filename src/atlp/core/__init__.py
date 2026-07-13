"""
    atlp.core package

    This package contains all essential functions to manage, retrieve, and modify data from the teleoperation data.

    There are three subpackages:
        1. atlp.core.teleop_reader
            Handle retrieving data from the teleoperation data, which includes
                - joint states (from data.json)
                - videos (from *.mp4)
                - frequency and duration (from report.txt)
        2. atlp.core.interface
            Handle retrieving/modifying/managing ATLP fields. Currently use header.json to store all ATLP field values.
        3. atlp.core.dataset_manager
            Additional quality-of-life functions that use the functions from .teleop_reader and .interface.

    Note:
        - User is encouraged to modify this code to match the robot data they have. (modifying atlp.core.teleop_reader)
        - Moreover, if the user finds that header.json is too slow for a larger dataset, 
        they are encouraged to upgrade the ATLP fields storage. (modifying atlp.core.interface)

    Warning:
        Because the atlp.core is referenced by every other subpackage, please modify the code with caution.
        Namely, every mandatory input and output of a function should be properly maintained when you edit the code.
"""

from .interface import (
    set_root,
    populate_header,
    show_statistics,
    list_datapoints,
    modify_datapoint,
    get_datapoint,
    set_enable_actual_field,
    tag_get,
    tag_remove,
    get_root_directory,
    load_data,
    tag_add,
)

from .teleop_reader import (
    get_joint_states,
    get_all_joint_states,
    get_video_frames,
    get_video_duration,
)

from .dataset_manager import (
    display_instruction,
    reset_all,
    get_single_datapoint,
    filter,
)