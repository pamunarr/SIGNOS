# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:51:14 2025

@author: Pablo Munarriz Senosiain
"""

import bpy
import numpy as np
from pickle import load
from arm_movement import max_left_shoulder_rotation, max_right_shoulder_rotation, landmarks2arm_position, arm_position2landmarks
from config import pi
from mathutils import Vector, Matrix


bone_conexions_list = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [4, 6]]
bone_names = ["l_hip", "r_hip", "d_backbone", "u_backbone", "l_shoulder", "r_shoulder"]
bone_parents_list = [-1, -1, -1, 2, 3, 3]

bone_conexions_list.extend([[i, i+2] for i in range(5, 9)])
bone_names.extend(["%s_%s" % (side, part) for part in ["upperarm", "forearm"] for side in ["l", "r"]])
bone_parents_list.extend(range(4, 8))

bone_conexions_list.extend([[9+side+2*(i!=0)*(4*finger+i), 9+side+2*(4*finger+i+1)] for finger in range(5) for i in range(4) for side in range(2)])
bone_names.extend(["%s_%s_%d" % (side, finger_name, i)
                   for finger_name in ["thumb", "index", "middle", "ring", "pinky"] for i in range(4) for side in ["l", "r"]])
bone_parents_list.extend([8+side+2*(i!=0)*(4*finger+i) for finger in range(5) for i in range(4) for side in range(2)])

bone_conexions = np.array(bone_conexions_list, dtype=np.uint8)
bone_parents = np.array(bone_parents_list, dtype=np.int8)


trunk_landmarks_avatar = np.array([
    [0.15, 0., 0.],
    [-0.15, 0., 0.],
    [0.2, -0.5, 0.],
    [-0.2, -0.5, 0.]
], dtype=np.float64)

left_finger_directions = np.array([
    [[0, -np.sin(0.31*pi), np.cos(0.31*pi)],
     [0, -np.sin(0.09*pi), np.cos(0.09*pi)],
     [0, -np.sin(0.02*pi), np.cos(0.02*pi)],
     [0, np.sin(0.03*pi), np.cos(0.03*pi)],
     [0, np.sin(0.11*pi), np.cos(0.11*pi)]],
    [[0, -np.sin(0.17*pi), np.cos(0.17*pi)],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]],
    [[0, -np.sin(0.17*pi), np.cos(0.17*pi)],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]],
    [[0, -np.sin(0.17*pi), np.cos(0.17*pi)],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]
], dtype=np.float64)
right_finger_directions = np.array([
    [[0, np.sin(0.31*pi), np.cos(0.31*pi)],
     [0, np.sin(0.09*pi), np.cos(0.09*pi)],
     [0, np.sin(0.02*pi), np.cos(0.02*pi)],
     [0, -np.sin(0.03*pi), np.cos(0.03*pi)],
     [0, -np.sin(0.11*pi), np.cos(0.11*pi)]],
    [[0, np.sin(0.17*pi), np.cos(0.17*pi)],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]],
    [[0, np.sin(0.17*pi), np.cos(0.17*pi)],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]],
    [[0, np.sin(0.17*pi), np.cos(0.17*pi)],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1],
     [0, 0, 1]]
], dtype=np.float64)
left_finger_lengths = np.array([
    [0.03, 0.103, 0.103, 0.1, 0.09],
    [0.045, 0.031, 0.035, 0.032, 0.025],
    [0.032, 0.024, 0.026, 0.023, 0.017],
    [0.033, 0.024, 0.025, 0.025, 0.023]
], dtype=np.float64).reshape(4, 5, 1)
right_finger_lengths = left_finger_lengths.copy()

arms_position_rest = {
    "left_shoulder_direction": np.array([0, 1, 0], dtype=np.float64),
    "left_upperarm_length": np.float64(0.33),
    "left_shoulder_rotation": pi/2.,
    "left_elbow_angle": pi,
    "left_forearm_length": np.float64(0.27),
    "left_elbow_rotation": np.float64(0),
    "left_wrist_rotation": np.float64(0),
    "left_wrist_inclination": np.float64(0),
    "left_finger_directions": left_finger_directions,
    "left_finger_lengths": left_finger_lengths,

    "right_shoulder_direction": np.array([0, 1, 0], dtype=np.float64),
    "right_upperarm_length": np.float64(0.33),
    "right_shoulder_rotation": pi/2.,
    "right_elbow_angle": pi,
    "right_forearm_length": np.float64(0.27),
    "right_elbow_rotation": np.float64(0),
    "right_wrist_rotation": np.float64(0),
    "right_wrist_inclination": np.float64(0),
    "right_finger_directions": right_finger_directions,
    "right_finger_lengths": right_finger_lengths
}

half_landmarks_rest = arm_position2landmarks(arms_position_rest, trunk_landmarks_avatar, 1, 1)


def get_half_landmarks(data, letter, frame_start, frame_stop=None):
    if not frame_stop:
        frame_stop = frame_start + 1
    half_landmarks = np.empty((frame_stop - frame_start, 48, 3), dtype=np.float64)
    half_landmarks[:, :6] = data[letter]["landmarks"]["pose"][frame_start:frame_stop, [23, 24] + list(range(11, 15)), :-1]
    half_landmarks[:, 6::2] = data[letter]["landmarks"]["left_hand"][frame_start:frame_stop]
    half_landmarks[:, 6::2, -1] += data[letter]["landmarks"]["pose"][frame_start:frame_stop, 15, 2].reshape(-1, 1)
    half_landmarks[:, 7::2] = data[letter]["landmarks"]["right_hand"][frame_start:frame_stop]
    half_landmarks[:, 7::2, -1] += data[letter]["landmarks"]["pose"][frame_start:frame_stop, 16, 2].reshape(-1, 1)
    return half_landmarks.squeeze()

def get_half_landmarks_maksym(data, letter, version, frame_start, frame_stop=None):
    if not frame_stop:
        frame_stop = frame_start + 1
    half_landmarks = np.empty((frame_stop - frame_start, 48, 3), dtype=np.float64)
    half_landmarks[:, :6] = data[letter]["poses_3d" + version][frame_start:frame_stop, [7, 6, 3, 0, 4, 1]]
    half_landmarks[:, 6::2] = np.where(
        data[letter]['success_left'][frame_start:frame_stop].reshape(-1, 1, 1),
        data[letter]["poses_3d" + version][frame_start:frame_stop, 40:], np.nan
    )
    half_landmarks[:, 7::2] = np.where(
        data[letter]['success_right'][frame_start:frame_stop].reshape(-1, 1, 1),
        data[letter]["poses_3d" + version][frame_start:frame_stop, 19:40], np.nan
    )
    return half_landmarks.squeeze()

def get_half_landmarks_maksym_(word, version, frame_start, frame_stop=None):
    word_path = "../Datos/Procesados/results_3d/palabras/%s.pkl" % word
    with open(word_path, "rb") as f:
        data = load(f)

    if not frame_stop:
        frame_stop = frame_start + 1
    half_landmarks = np.empty((frame_stop - frame_start, 48, 3), dtype=np.float64)
    half_landmarks[:, :6] = data["poses_3d" + version][frame_start:frame_stop, [7, 6, 3, 0, 4, 1]]
    half_landmarks[:, 6::2] = np.where(
        data["success_left"][frame_start:frame_stop].reshape(-1, 1, 1),
        data["poses_3d" + version][frame_start:frame_stop, 40:], np.nan
    )
    half_landmarks[:, 7::2] = np.where(
        data["success_right"][frame_start:frame_stop].reshape(-1, 1, 1),
        data["poses_3d" + version][frame_start:frame_stop, 19:40], np.nan
    )
    return half_landmarks.squeeze()

def get_shoulder_basis(half_landmarks):
    shoulder_basis = np.empty((2, 3, 3), dtype=np.float64)

    shoulder_basis[0, :, 0] = half_landmarks[2] - half_landmarks[3]
    shoulder_basis[0, :, 2] = np.cross(shoulder_basis[0, :, 0], half_landmarks[0] - half_landmarks[2])
    shoulder_basis[0, :, 1] = np.cross(shoulder_basis[0, :, 2], shoulder_basis[0, :, 0])

    shoulder_basis[1, :, 0] = half_landmarks[3] - half_landmarks[2]
    shoulder_basis[1, :, 2] = np.cross(shoulder_basis[1, :, 0], half_landmarks[1] - half_landmarks[3])
    shoulder_basis[1, :, 1] = np.cross(shoulder_basis[1, :, 2], shoulder_basis[1, :, 0])

    shoulder_basis /= np.linalg.norm(shoulder_basis, axis=1, keepdims=True)
    return shoulder_basis

def get_hand_basis(half_landmarks):
    hand_basis = np.empty((2, 3, 3), dtype=np.float64)

    hand_basis[0, :, 0] = np.cross(half_landmarks[40] - half_landmarks[6], half_landmarks[16] - half_landmarks[6])
    hand_basis[0, :, 2] = (half_landmarks[40] + half_landmarks[16])/2. - half_landmarks[6]
    hand_basis[0, :, 1] = np.cross(hand_basis[0, :, 2], hand_basis[0, :, 0])

    hand_basis[1, :, 0] = np.cross(half_landmarks[17] - half_landmarks[7], half_landmarks[41] - half_landmarks[7])
    hand_basis[1, :, 2] = (half_landmarks[41] + half_landmarks[17])/2. - half_landmarks[7]
    hand_basis[1, :, 1] = np.cross(hand_basis[1, :, 2], hand_basis[1, :, 0])

    hand_basis /= np.linalg.norm(hand_basis, axis=1, keepdims=True)
    return hand_basis

def get_rot_angles_v(v1, v2, v3):
    if np.isclose(v2, 1):
        beta = 0.
        gamma = 0.
    else:
        if np.isclose(v3*v3, 1):
            beta = np.sign(v3) * pi/2.
            gamma = 0.
        else:
            gamma = np.arctan2(-v1, v2)
            beta = np.arctan2(v3, -v1/np.sin(gamma)) if np.isclose(v2, 0) else np.arctan2(v3, v2/np.cos(gamma))
    return beta, gamma

def imaginary_rotation(beta, gamma, bone_matrix, half_landmarks, base_index, side):
    rotation_matrix = np.array([
        [np.cos(gamma), np.sin(gamma), 0],
        [- np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma), np.sin(beta)],
        [np.sin(beta) * np.sin(gamma), - np.sin(beta) * np.cos(gamma), np.cos(beta)]
    ], dtype=np.float64)
    bone_matrix_np = np.array(bone_matrix, dtype=np.float64)
    moving_vectors = (half_landmarks[2*np.arange(base_index//2 + 1, 24)+side] - half_landmarks[base_index+side])[:, [0, 2, 1]] * np.array([1, 1, -1])

    half_landmarks[2*np.arange(base_index//2 + 1, 24)+side] = (moving_vectors @ bone_matrix_np @ rotation_matrix @ bone_matrix_np.T
                                                              )[:, [0, 2, 1]] * np.array([1, -1, 1]) + half_landmarks[base_index+side]

def get_wrist_rotation_matrix(arms_position, side, inverse):
    w = ["left", "right"][side]
    sign = 1 if inverse else -1

    a = np.cos(arms_position["%s_wrist_inclination" % w])
    b = np.sin(arms_position["%s_wrist_inclination" % w])
    c = np.cos(arms_position["%s_wrist_rotation" % w])
    d = np.sin(arms_position["%s_wrist_rotation" % w])

    return Matrix([
        [1.+(a-1.)*c*c, -sign*b*c, (1.-a)*c*d],
        [sign*b*c, a, -sign*b*d],
        [(1.-a)*c*d, sign*b*d, 1.+(a-1.)*d*d]
    ])

def get_arms_position(half_landmarks, frame_dimensions, arms_position_base):
    arms_position = landmarks2arm_position(half_landmarks, *frame_dimensions)
    for k, v in arms_position.items():
        if np.isnan(v).any():
            arms_position[k] = arms_position_base[k]
    return arms_position

def isiterable(x):
    return "__iter__" in dir(x)

def is_dynamic_pose(x):
    return isiterable(x) and all(map(lambda y : y.shape==(48, 3), x))


def move_arms_blender(
    armature_filepath, armature_name, half_landmarks_armature, arms_position_armature,
    list_half_landmarks, list_frame_dimensions, list_time_movement, list_time_position,
    fps, animation_filepath
):
    # Obtain the armature
    bpy.ops.wm.open_mainfile(filepath=armature_filepath)
    armature = bpy.data.objects.get(armature_name)
    bpy.context.view_layer.objects.active = armature
    arms_bones_names = bone_names[6:]

    def set_pose(half_world_landmarks_0, arms_position_0, arms_position_1):
        half_world_landmarks_ = half_world_landmarks_0.copy()
        arms_position_ = arms_position_0.copy()

        shoulder_basis = get_shoulder_basis(half_world_landmarks_)

        def upperarm_rotation():
            for side in range(2):
                l = ["l", "r"][side]
                w = ["left", "right"][side]
            
                bone_name = "%s_upperarm" % l
                bone = armature.pose.bones[bone_name]
                bone.rotation_mode = 'YXZ'
            
                upperarm_image = arms_position_1["%s_shoulder_direction" % w] @ shoulder_basis[side].T
                v1, v2, v3 = bone.matrix.transposed().to_3x3() @ Vector(upperarm_image[[0, 2, 1]] * np.array([1, 1, -1]))
            
                beta, gamma = get_rot_angles_v(v1, v2, v3)
                if np.isclose(v2, 1):
                    alpha = (arms_position_1["%s_shoulder_rotation" % w] - arms_position_["%s_shoulder_rotation" % w]) * (1 - 2*side)
                else:
                    imaginary_rotation(beta, gamma, bone.matrix.to_3x3(), half_world_landmarks_, 2, side)
            
                    if np.isclose(arms_position_["%s_elbow_angle" % w], pi):
                        aux_position = landmarks2arm_position(half_world_landmarks_, 1, 1)
                        shoulder_rotation = (aux_position["%s_shoulder_rotation" % w] +
                                             aux_position["%s_elbow_rotation" % w] - arms_position_["%s_elbow_rotation" % w])
                        alpha = (arms_position_1["%s_shoulder_rotation" % w] - shoulder_rotation) * (1 - 2*side)
                    else:
                        alpha = (arms_position_1["%s_shoulder_rotation" % w] -
                                 landmarks2arm_position(half_world_landmarks_, 1, 1)["%s_shoulder_rotation" % w]) * (1 - 2*side)
                bone.rotation_euler = (beta, alpha, gamma)

        def forearm_rotation():
            for side in range(2):
                l = ["l", "r"][side]
                w = ["left", "right"][side]

                bone_name = "%s_forearm" % l
                bone = armature.pose.bones[bone_name]
                bone.rotation_mode = 'YXZ'

                if np.isclose(arms_position_["%s_elbow_angle" % w], arms_position_1["%s_elbow_angle" % w]):
                    alpha = (arms_position_1["%s_elbow_rotation" % w] - arms_position_["%s_elbow_rotation" % w]) * (1 - 2*side)
                    beta = 0.
                    gamma = 0.
                else:
                    bone_matrix = bone.matrix.to_3x3()

                    v2 = (arms_position_["%s_shoulder_direction" % w] @ shoulder_basis[side].T)[[0, 2, 1]] * np.array([1, 1, -1])
                    if np.isclose(arms_position_["%s_elbow_angle" % w], pi):
                        max_shoulder_rotation = [max_left_shoulder_rotation, max_right_shoulder_rotation][side]
                        arm_plane_normal_vector_max_rotation = (max_shoulder_rotation(arms_position_["%s_shoulder_direction" % w]) @
                                                                shoulder_basis[side].T)[[0, 2, 1]] * np.array([1, 1, -1])
                        aux_vector = np.cross(v2, arm_plane_normal_vector_max_rotation) * (1 - 2*side)
                        arm_plane_normal_vector = (np.cos(arms_position_["%s_shoulder_rotation" % w]) * arm_plane_normal_vector_max_rotation +
                                                   np.sin(arms_position_["%s_shoulder_rotation" % w]) * aux_vector)
                        v1 = - arm_plane_normal_vector
                    else:
                        forearm_direction = (half_world_landmarks_[6+side] - half_world_landmarks_[4+side])[[0, 2, 1]] * np.array([1, 1, -1])
                        forearm_direction /= np.linalg.norm(forearm_direction)
                        v1 = np.cross(forearm_direction, v2)
                        v1 /= np.linalg.norm(v1)
                    v3 = np.cross(v1, v2)
                    parent_matrix = Matrix(np.array([v1, v2, v3]).T)

                    beta_parent = arms_position_1["%s_elbow_angle" % w] - arms_position_["%s_elbow_angle" % w]
                    rotation_matrix_parent = Matrix([
                        [1, 0, 0],
                        [0, np.cos(beta_parent), - np.sin(beta_parent)],
                        [0, np.sin(beta_parent), np.cos(beta_parent)]
                    ])
                    rotation_matrix_bone = bone_matrix.transposed() @ parent_matrix @ rotation_matrix_parent @ parent_matrix.transposed() @ bone_matrix
                    beta, _, gamma = rotation_matrix_bone.to_euler("YXZ")

                    imaginary_rotation(beta, gamma, bone_matrix, half_world_landmarks_, 4, side)

                    alpha = (arms_position_1["%s_elbow_rotation" % w] -
                             landmarks2arm_position(half_world_landmarks_, 1, 1)["%s_elbow_rotation" % w]) * (1 - 2*side)
                bone.rotation_euler = (beta, alpha, gamma)

        def wrist_rotation():
            hand_basis = get_hand_basis(half_world_landmarks_)
            for side in range(2):
                l = ["l", "r"][side]
                w = ["left", "right"][side]

                forearm_direction = half_world_landmarks_[6+side] - half_world_landmarks_[4+side]
                forearm_direction /= np.linalg.norm(forearm_direction)
                if np.isclose(arms_position_["%s_elbow_angle" % w], pi):
                    max_shoulder_rotation = [max_left_shoulder_rotation, max_right_shoulder_rotation][side]
                    arm_rotation = arms_position_["%s_shoulder_rotation" % w] + arms_position_["%s_elbow_rotation" % w] + pi/2.
                    aux_vector = max_shoulder_rotation(arms_position_["%s_shoulder_direction" % w]) @ shoulder_basis[side].T
                    palm_normal_vector_max_arm_rotation = np.cross(aux_vector, forearm_direction)
                    palm_normal_vector_no_wrist_inclination = (np.cos(arm_rotation) * palm_normal_vector_max_arm_rotation +
                                                               (1 - 2*side) * np.sin(arm_rotation) * aux_vector)
                else:
                    palm_normal_vector_no_elbow_rotation = np.cross(arms_position_["%s_shoulder_direction" % w] @ shoulder_basis[side].T,
                                                                    forearm_direction) * (1 - 2*side)
                    palm_normal_vector_no_elbow_rotation /= np.linalg.norm(palm_normal_vector_no_elbow_rotation)
                    aux_vector = np.cross(forearm_direction, palm_normal_vector_no_elbow_rotation)
                    palm_normal_vector_no_wrist_inclination = (np.cos(arms_position_["%s_elbow_rotation" % w]) * palm_normal_vector_no_elbow_rotation +
                                                               (1 - 2*side) * np.sin(arms_position_["%s_elbow_rotation" % w]) * aux_vector)
                v1 = palm_normal_vector_no_wrist_inclination[[0, 2, 1]] * np.array([1, 1, -1])
                v3 = forearm_direction[[0, 2, 1]] * np.array([1, 1, -1])
                v2 = np.cross(v3, v1)
                parent_matrix = Matrix(np.array([v1, v3, -v2]).T)

                rotation_matrix_parent_0 = get_wrist_rotation_matrix(arms_position_, side, True)
                rotation_matrix_parent_1 = get_wrist_rotation_matrix(arms_position_1, side, False)
                rotation_matrix = parent_matrix @ rotation_matrix_parent_1 @ rotation_matrix_parent_0 @ parent_matrix.transposed()

                bone_names_ = ["%s_%s_0" % (l, finger) for finger in ["index", "middle", "ring", "pinky"]]
                for bone_name in bone_names_:
                    bone = armature.pose.bones[bone_name]
                    bone.rotation_mode = 'YXZ'
        
                    bone_matrix = bone.matrix.to_3x3()
                    rotation_matrix_bone = bone_matrix.transposed() @ rotation_matrix @ bone_matrix
                    bone.rotation_euler = rotation_matrix_bone.to_euler("YXZ")

                bone_name = "%s_thumb_0" % l
                bone = armature.pose.bones[bone_name]
                bone.rotation_mode = 'YXZ'
                bone_matrix = bone.matrix.to_3x3()

                thumb_0_direction = arms_position_1["%s_finger_directions" % w][0, 0] @ hand_basis[side].T
                v1, v2, v3 = bone_matrix.transposed() @ Vector(thumb_0_direction[[0, 2, 1]] * np.array([1, 1, -1]))
                beta, gamma = get_rot_angles_v(v1, v2, v3)

                rotation_matrix_bone = (bone_matrix.transposed() @ rotation_matrix @ bone_matrix @
                                        Matrix().Rotation(gamma, 3, 'Z') @ Matrix().Rotation(beta, 3, 'X'))
                bone.rotation_euler = rotation_matrix_bone.to_euler("YXZ")

        def update_position(check_arm_extension=False):
            bpy.context.view_layer.update()
            for bone_number in range(6, 50):
                bone = armature.pose.bones[bone_names[bone_number]]
                half_world_landmarks_[bone_number-2] = np.array(bone.tail, dtype=np.float64)[[0, 2, 1]] * np.array([1, -1, 1])

            if check_arm_extension:
                left_arm_is_extended = np.isclose(arms_position_["left_elbow_angle"], pi)
                right_arm_is_extended = np.isclose(arms_position_["right_elbow_angle"], pi)
                left_elbow_rotation = arms_position_["left_elbow_rotation"]
                right_elbow_rotation = arms_position_["right_elbow_rotation"]
                arms_position_.update(landmarks2arm_position(half_world_landmarks_, 1, 1))
                if left_arm_is_extended:
                    arms_position_["left_shoulder_rotation"] = (arms_position_["left_shoulder_rotation"] +
                                                                arms_position_["left_elbow_rotation"] - left_elbow_rotation)
                    arms_position_["left_elbow_rotation"] = left_elbow_rotation
                if right_arm_is_extended:
                    arms_position_["right_shoulder_rotation"] = (arms_position_["right_shoulder_rotation"] +
                                                                 arms_position_["right_elbow_rotation"] - right_elbow_rotation)
                    arms_position_["right_elbow_rotation"] = right_elbow_rotation
            else:
                arms_position_.update(landmarks2arm_position(half_world_landmarks_, 1, 1))

        # Upperarm direction and rotation
        upperarm_rotation()
        update_position(True)

        # Forearm direction and rotation
        forearm_rotation()
        update_position()

        # Wrist rotation and inclination, thumb direction
        wrist_rotation()
        update_position()

        # Fingers
        hand_basis = get_hand_basis(half_world_landmarks_)

        for phalanx in range(1, 4):
            for side in range(2):
                l = ["l", "r"][side]
                w = ["left", "right"][side]

                bone_names_ = ["%s_%s_%d" % (l, finger, phalanx) for finger in ["thumb", "index", "middle", "ring", "pinky"]]

                finger_directions_image = (arms_position_1["%s_finger_directions" % w][phalanx] @ hand_basis[side].T
                                          )[:, [0, 2, 1]] * np.array([1, 1, -1])
                for finger in range(5):
                    bone = armature.pose.bones[bone_names_[finger]]
                    bone.rotation_mode = 'YXZ'
                    
                    v1, v2, v3 = bone.matrix.transposed().to_3x3() @ Vector(finger_directions_image[finger])
                    beta, gamma = get_rot_angles_v(v1, v2, v3)
                    bone.rotation_euler = (beta, 0, gamma)
            update_position()

    def insert_pose(frame):
        for bone_name in arms_bones_names:
            bone = armature.pose.bones[bone_name]
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)

    def clear_rotations():
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.rot_clear()

    def set_pose_in_interval(half_landmarks, frame_dimensions, T, time_position):
        arms_position = get_arms_position(half_landmarks, frame_dimensions, arms_position_armature)
        set_pose(half_landmarks_armature, arms_position_armature, arms_position)
        insert_pose(int(T*fps)) # Set the pose in the first frame
        T += time_position
        insert_pose(int(T*fps)) # Set the pose in the last frame
        clear_rotations()
        return T

    def set_dynamic_pose(dynamic_half_landmarks, frame_dimensions, T0):
        frame = int(T0*fps)
        for half_landmarks in dynamic_half_landmarks:
            arms_position = get_arms_position(half_landmarks, frame_dimensions, arms_position_armature)
            set_pose(half_landmarks_armature, arms_position_armature, arms_position)
            insert_pose(frame)
            frame += 1
            clear_rotations()
        return frame/fps

    # Start animation
    bpy.ops.object.mode_set(mode='POSE')

    assert isiterable(list_half_landmarks), "`list_half_landmarks` must be an iterable."
    n_dynamic_positions = sum(map(is_dynamic_pose, list_half_landmarks))

    if isinstance(list_time_movement, (int, float)):
        list_time_movement = [list_time_movement] * (len(list_half_landmarks) - 1)
    else:
        assert isiterable(list_time_movement), "`list_time_movement` must be a number or an iterable."
        assert len(list_time_movement)==len(list_half_landmarks) - 1, "`list_time_movement` must have one element less than `list_half_landmarks`."

    if isinstance(list_time_position, (int, float)):
        list_time_position = [list_time_position] * (len(list_half_landmarks) - n_dynamic_positions)
    else:
        assert isiterable(list_time_position), "`list_time_position` must be a number or an iterable."
        error_text = "`list_time_position` must have as many elements as no-dynamic poses in `list_half_landmarks`."
        assert len(list_time_position)==len(list_half_landmarks) - n_dynamic_positions, error_text

    if isinstance(list_frame_dimensions, tuple):
        list_frame_dimensions = [list_frame_dimensions] * len(list_half_landmarks)
    else:
        assert isiterable(list_frame_dimensions), "`list_frame_dimensions` must be a number or an iterable."
        assert len(list_time_position)==len(list_half_landmarks), "`list_frame_dimensions` must have as many elements as `list_half_landmarks`."

    T = 0
    for half_landmarks in list_half_landmarks:
        if is_dynamic_pose(half_landmarks):
            T = set_dynamic_pose(half_landmarks, list_frame_dimensions.pop(0), T)
        else:
            T = set_pose_in_interval(half_landmarks, list_frame_dimensions.pop(0), T, list_time_position.pop(0))
    
        if list_time_movement:
            T += list_time_movement.pop(0)

    # End and save animation
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.save_as_mainfile(filepath=animation_filepath)



def tokenize_word(word):
    """
    Given a word using the spanish alphabet, returns a list with the letters of the word.
    It includes the letters "CH", "LL", "Ñ" and "RR".

    Input: word (string)
    Output: token_list (list of strings)
    """
    # Define the alphabet
    alphabet = [
        'A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'LL', 'M',
        'N', 'Ñ', 'O', 'P', 'Q', 'R', 'RR', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    # Write the word in capital letters
    word = word.upper()

    # Verify that all the letters of the word are in the alphabet
    if not all(map(lambda char : char in alphabet, word)):
        raise Exception("Some letter of the word is not in the alphabet.")

    # Initialize the token list
    token_list = []
    # Go over the word adding tokens to the list
    for letter in word:
        # If the token is "CH", "LL" or "RR", we have to remove the last letter from the list and add the complete token
        if token_list and any([token_list[-1]=="C" and letter=="H", token_list[-1]=="L" and letter=="L", token_list[-1]=="R" and letter=="R"]):
            token_list.append(token_list.pop() + letter)
        # If the letter is "Ñ", add "N_"
        elif letter=="Ñ":
            token_list.append("N_")
        # Otherwhise, just add the letter
        else:
            token_list.append(letter)
    return token_list

frames_letter = {}
with open("frame_letras_spread.txt", 'r') as f:
    for line in f.readlines():
        datos = line[:-1].split(' ')
        letter = datos[0]
        frames = list(map(int, datos[1].split('-')))
        frames_letter[letter] = frames


def spell_word(data, word, animation_filepath=None, version="_filtered", start_rest=True, end_rest=True,
               armature_filepath="../Datos/Procesados/avatar_armature.blend", armature_name="Armature",
               half_landmarks_armature=half_landmarks_rest, arms_position_armature=arms_position_rest,
               list_frame_dimensions=(1, 1), list_time_movement=1, list_time_position=0.5, fps=24):
    list_half_landmarks = []
    if start_rest:
        list_half_landmarks.append(half_landmarks_rest)
    for letter in tokenize_word(word):
        frames = frames_letter[letter]
        list_half_landmarks.append(get_half_landmarks_maksym(data, letter, version, *frames))
    if end_rest:
        list_half_landmarks.append(half_landmarks_rest)

    if not animation_filepath:
        animation_filepath = word + ".blend"
    move_arms_blender(
        armature_filepath, armature_name, half_landmarks_armature, arms_position_armature,
        list_half_landmarks, list_frame_dimensions, list_time_movement, list_time_position, fps, animation_filepath
    )