import pinocchio as pin
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from .interface import (
    get_all_joint_states,
    get_video_duration,
    filter,
    load_data,
    write_data,
    modify_datapoint,
    get_single_datapoint,
)

_model_bin_path = "model.bin"
_collision_bin_path = "collision_model.bin"

# file_path = 'data.json'
# report_file_path = 'report.txt'

# mjcf_path = Path("~/auto-task-labelling-pipeline/src/atlp/galbot_one_golf_collision_only.xml").expanduser()
mjcf_path = Path("/home/o25141/galbot-sim-ioai/physics_sim_edu/assets/synthnova_assets/robots/galbot_one_foxtrot_description_simplified/galbot_one_foxtrot.xml")


# left_gripper and right_gripper do not appear in data.json,
# but are made by convert_to_np_array for convenience 
joint_lim_dict = {
    "leg_joint1"        :   (0.0, 0.9374),
    "leg_joint2"        :   (0.0, 2.5847),
    "leg_joint3"        :   (0.0, 2.3262),
    "leg_joint4"        :   (-1.5906, 1.5906),
    "leg_joint5"        :   (-0.1645, 0.1645),
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

def _get_idx(model, joint_name):
    """
        Return index of the joint name from the given pinocchio model

        Args:
            model: 
            joint_name:
    """
    return model.joints[model.getJointId(joint_name)].idx_q

def _build_q(model, all_joint_arrays, t):
    """
        Populate the model with the all_joint_arrays at time t

        Args:
            model:
            all_joint_arrays:
            t:
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

    # q[_get_idx(model, "left_gripper_joint")] = all_joint_arrays[27, t]
    # q[_get_idx(model, "right_gripper_joint")] = all_joint_arrays[28, t]

    return q

def _remove_srdf_disabled_collisions(model, collision_model, srdf_path):
    """Parse SRDF and remove all <disable_collisions> pairs by link name."""
    tree = ET.parse(srdf_path)
    root = tree.getroot()

    for dc in root.findall("disable_collisions"):
        link1 = dc.get("link1")
        link2 = dc.get("link2")
        _remove_collision_pairs_by_link_names(model, collision_model, link1, link2)

def _get_joint_idx_for_link(model, link_name):
    """Get the parent joint index for a link, searching both joint names and frame names."""
    # 1. Try direct joint name match
    joint_idx = {model.names[i]: i for i in range(model.njoints)}
    if link_name in joint_idx:
        return joint_idx[link_name]

    # 2. Fall back: find link as a frame and return its parentJoint
    for frame in model.frames:
        if frame.name == link_name:
            return frame.parentJoint

    return None


def _remove_collision_pairs_by_link_names(model, collision_model, link1_name, link2_name):
    j1 = _get_joint_idx_for_link(model, link1_name)
    j2 = _get_joint_idx_for_link(model, link2_name)

    if j1 is None or j2 is None:
        print(f"Warning: could not resolve '{link1_name}' or '{link2_name}' to a joint")
        return

    geoms1 = {i for i, g in enumerate(collision_model.geometryObjects) if g.parentJoint == j1}
    geoms2 = {i for i, g in enumerate(collision_model.geometryObjects) if g.parentJoint == j2}

    pairs_to_remove = [
        pair for pair in collision_model.collisionPairs
        if (pair.first in geoms1 and pair.second in geoms2) or
           (pair.first in geoms2 and pair.second in geoms1)
    ]
    for pair in pairs_to_remove:
        collision_model.removeCollisionPair(pair)

def is_self_collision(
    all_joint_arrays,
    distance_threshold : float = 0.02,
    model = None,
    collision_model = None,
    is_calculate_distance = False,
    sampling_frequency : float = 10, # Hz
    frequency : float = 30, # Hz
    stop_at_first_collision : bool = True, # Only meaningful if is_calculate_distance=True
    is_verbose : bool = False,
) -> tuple[bool, int]:
    """
        Check whether the given all_joint_arrays have collision or not
    
        Returns:
            (is_self_collision, collision_timestamp) where
                is_self_collision is bool
                collision_timestamp is the timestamp that there is collision, -1 if there is no collision
        ..todo::
            - properly setup the paths to model.bin and collision_model.bin
            - recompute .bin files so that the meshes are referenced correctly under the package directory
    """
    delta_n = int(frequency/sampling_frequency)

    if (model is None) or (collision_model is None):
        model = pin.Model()
        model.loadFromBinary(_model_bin_path)

        collision_model = pin.GeometryModel()
        collision_model.loadFromBinary(_collision_bin_path)
        
        # still needed if you use named configs later
        # pin.loadReferenceConfigurations(model, srdf_path, verbose=False)
        
        data = model.createData()
        collision_data = collision_model.createData()
    
    n = all_joint_arrays.shape[1]
    is_collision = False
    
    if is_calculate_distance:
        # Calculate distance
        collision_timestamp = -1
        for t in range(0, n, delta_n):
            q = _build_q(model, all_joint_arrays, t)
            pin.forwardKinematics(model, data, q)          # <-- add this
            pin.computeDistances(model, data, collision_model, collision_data, q)
        
            for k, cp in enumerate(collision_model.collisionPairs):
                dr = collision_data.distanceResults[k]
                if dr.min_distance < distance_threshold:
                    name1 = collision_model.geometryObjects[cp.first].name
                    name2 = collision_model.geometryObjects[cp.second].name
                    id1 = cp.first
                    id2 = cp.second
                    if is_verbose:
                        print(f"Collision: {name1}|{id1} <-> {name2}|{id2}: {dr.min_distance:.4f} m at t={t}")
                    if stop_at_first_collision: return True, t
                    elif not is_collision:
                        is_collision = True
                        collision_timestamp = t
        return is_collision,collision_timestamp
    else:
        # Only booleans
        
        for t in range(0, n, delta_n):
            # print(f"Computing timestamp {t}")
            q = _build_q(model, all_joint_arrays, t)
            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, collision_model, collision_data)
        
            # Set by index — modifies the actual object, not a copy
            for k in range(len(collision_model.collisionPairs)):
                collision_data.collisionRequests[k].security_margin = distance_threshold
                
            # 2. Check all pairs
            is_collision = pin.computeCollisions(
            collision_model, collision_data,
            stop_at_first_collision=True   # True = faster, stops early
            )
        
            if is_collision: 
                if is_verbose:
                    print(f"Collision detected at timestamp {t}")
                return True, t
        return False, -1

def is_self_collision_from_datapoint(    
    datapoint : str,
    frequency : float = 30, # Hz
    distance_threshold : float = 0.02,
    model = None,
    collision_model = None,
    is_calculate_distance = False,
    sampling_frequency : float = 10, # Hz
    stop_at_first_collision : bool = True, # Only meaningful if is_calculate_distance=True
    is_verbose : bool = False,
) -> tuple[bool, int]:
    total_time = get_video_duration(datapoint)
    _, all_joint_arrays , *_ = get_all_joint_states(
                                                    datapoint,
                                                    frequency=frequency,
                                                    total_time=total_time,
                                                )
    return is_self_collision(
        all_joint_arrays=all_joint_arrays,
        distance_threshold=distance_threshold,
        model=model,
        collision_model=collision_model,
        is_calculate_distance=is_calculate_distance,
        sampling_frequency=sampling_frequency,
        frequency=frequency,
        stop_at_first_collision=stop_at_first_collision,
        is_verbose=is_verbose,
    )

# OBSOLETE
def is_self_collision_obsolete(all_joint_arrays, distance_threshold=0.02):
    """
        Determine whether the given all_joint_arrays has any self collision or not.
        Does not update the value inside the header file
        Still not functionable.
    """
    model, _, collision_model, visual_model = pin.buildModelsFromMJCF(mjcf_path)
    data           = model.createData()
    collision_data = collision_model.createData()

    # Without SRDF, remove adjacent pairs manually:
    for pair in collision_model.collisionPairs:
        geom1 = collision_model.geometryObjects[pair.first]
        geom2 = collision_model.geometryObjects[pair.second]
        # Remove if they share a parent joint (adjacent)
        if geom1.parentJoint == geom2.parentJoint:
            collision_model.removeCollisionPair(pair)

    n = all_joint_arrays.shape[1]
    is_collision = False

    # for req in collision_data.collisionRequests:
    #     req.security_margin = 1  # applies to all pairs

    for t in range(0, n):

        q = _build_q(model, all_joint_arrays, t)

        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, collision_model, collision_data)

        # # Set margin AFTER updateGeometryPlacements, right before collision check
        # for req in collision_data.collisionRequests:
        #     req.security_margin = distance_threshold

        # Set by index — modifies the actual object, not a copy
        for k in range(len(collision_model.collisionPairs)):
            collision_data.collisionRequests[k].security_margin = distance_threshold
        
        # 2. Check all pairs
        is_collision = pin.computeCollisions(
            collision_model, collision_data,
            stop_at_first_collision=True   # True = faster, stops early
        )

        if is_collision: return True

    return False

def model_is_self_collision(
    datapoints : list[str] = [],
    frequency : float = 30,
    is_calculate_distance : bool = False,
    is_verbose : bool = False,
    use_dict : dict = dict()
) -> None | dict:
    if is_verbose:
        print("Starting self-collision checking")

    if len(use_dict) == 0:
        data = load_data()
    else:
        data = use_dict
    
    if len(datapoints) == 0:
        datapoints = filter(use_dict=data)
    for datapoint in datapoints:
        is_collision_checked = get_single_datapoint(
                                                    datapoint=datapoint,
                                                    main_field="motion",
                                                    field="is_self_collide",
                                                    use_dict=data,
                                                )
        if is_collision_checked is None:
            if is_verbose: print(f"Checking {datapoint}")
            is_collision, collision_timestamp = is_self_collision_from_datapoint(
                datapoint=datapoint,
                frequency=frequency,
                is_verbose=is_verbose,
                is_calculate_distance=is_calculate_distance,
            )
            if is_verbose: print(f"{datapoint}'s self-collision status is {is_collision} at t={collision_timestamp}")
            data = modify_datapoint(
                datapoint=datapoint,
                use_dict=data,
                motions={"is_self_collide": is_collision}
            )
        else:
            print(f"Skipping {datapoint} as it has already been checked.")

    if is_verbose:
        print("Finish self-collision checking.")
        
    if len(use_dict) == 0:
        write_data(data)
        return None
    return data
        

