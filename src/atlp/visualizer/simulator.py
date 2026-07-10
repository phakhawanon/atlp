import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import time

from ..modeller.motion.motion import (      
    mjcf_path,
    _build_q,
)
# import meshcat

# file_path = 'data.json'
# report_file_path = 'report.txt'

_SIMULATION_STEP_SIZE = 50

def simulate_joint_arrays(
    all_time_array,
    all_joint_arrays,
    time_step: int = _SIMULATION_STEP_SIZE
) -> None:
    """
        Open the browser and 3D simulate the given robot pose

        Args:
            all_time_array: The all_time_array obtained from teleop_reader.get_all_joint_states()
            all_joint_arrays: The all_joint_arrays obtained from teleop_reader.get_all_joint_states()
            time_step: Step size for the simulation, default to 50.
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