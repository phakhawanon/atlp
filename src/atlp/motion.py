import pinocchio as pin
import numpy as np
from pathlib import Path

# file_path = 'data.json'
# report_file_path = 'report.txt'

# mjcf_path = Path("~/auto-task-labelling-pipeline/src/atlp/galbot_one_golf_collision_only.xml").expanduser()
mjcf_path = Path("/home/o25141/galbot-sim-ioai/physics_sim_edu/assets/synthnova_assets/robots/galbot_one_foxtrot_description/galbot_one_foxtrot.xml")


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

    
def is_self_collision(all_joint_arrays, distance_threshold=0.02):
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
