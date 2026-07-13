"""
    Motion subpackage is a collection of functions which concern the analysis of motion to automatically model a datapoint.
"""

from .motion import (
    is_self_collision,
    is_self_collision_from_datapoint,
    model_is_self_collision,
)