# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 10:24:59 2024

@author: Pablo Munarriz Senosiain
"""

import numpy as np
from config import pi


def azimuth(x, y, z):
    """
    Given a unit 3D real vector, returns the azimuth of the vector.
    """
    return np.arctan2(y, x)
    # if np.isclose(x, 0):
    #     if np.isclose(y, 0):
    #         return np.float64(0)
    #     else:
    #         return np.sign(y) * pi / 2.
    # else:
    #     aux = np.arctan(y/x)
    #     if x>0:
    #         return aux
    #     elif y>0:
    #         return aux + pi
    #     else:
    #         return aux - pi

def colatitude(x, y, z):
    """
    Given a unit 3D real vector, returns the colatitude of the vector.
    """
    return np.arccos(np.clip(z, -1, 1))


def max_left_shoulder_rotation(left_shoulder_direction):
    """
    Given the left shoulder direction in the left shoulder basis, returns
    the normal vector of the left arm plane when it is maximally rotated.

    Input: array of shape (3,).

    Output: array of shape (3,).
    """
    x, y, z = left_shoulder_direction

    if np.isclose(y, 1):
        aux_vector = np.array([0, 0, -1], dtype=np.float64)
        angle = pi / 4.
    elif np.isclose(y, -1):
        aux_vector = np.array([0, 0, 1], dtype=np.float64)
        angle = 0.
    else:
        if np.isclose(y, 0):
            aux_vector = np.array([0, -1, 0], dtype=np.float64)
        else:
            aux_vector = np.sign(y) * np.array([x, y - 1./y, z], dtype=np.float64)
            aux_vector /= np.linalg.norm(aux_vector)

        tita = colatitude(x, z, y)
        fi = azimuth(x, z, y)
        if z<=0:
            if y>=0:
                angle = (fi + pi / 4.) * (2. * tita / pi - 1.)
            else:
                angle = (fi + pi / 2.) * (2. * tita / pi - 1.)
        else:
            if y>=0:
                angle = tita / 2. - fi - pi / 4.
            else:
                angle = - fi + (2. * fi + pi / 2.) * (2. * tita / pi - 1.)

    aux_normal = np.cross(aux_vector, left_shoulder_direction)
    left_elbow_direction_max_rotation = np.cos(angle) * aux_vector + np.sin(angle) * aux_normal
    
    return np.cross(left_shoulder_direction, left_elbow_direction_max_rotation)

def max_right_shoulder_rotation(right_shoulder_direction):
    """
    Given the right shoulder direction in the right shoulder basis, returns
    the normal vector of the right arm plane when it is maximally rotated.

    Input: array of shape (3,).

    Output: array of shape (3,).
    """
    x, y, z = right_shoulder_direction

    if np.isclose(y, 1):
        aux_vector = np.array([0, 0, 1], dtype=np.float64)
        angle = pi / 4.
    elif np.isclose(y, -1):
        aux_vector = np.array([0, 0, -1], dtype=np.float64)
        angle = 0.
    else:
        if np.isclose(y, 0):
            aux_vector = np.array([0, -1, 0], dtype=np.float64)
        else:
            aux_vector = np.sign(y) * np.array([x, y - 1./y, z], dtype=np.float64)
            aux_vector /= np.linalg.norm(aux_vector)

        tita = colatitude(x, z, y)
        fi = azimuth(x, z, y)
        if z>=0:
            if y>=0:
                angle = (pi / 4. - fi) * (2. * tita / pi - 1.)
            else:
                angle = (pi / 2. - fi) * (2. * tita / pi - 1.)
        else:
            if y>=0:
                angle = tita / 2. + fi - pi / 4.
            else:
                angle = fi + (pi / 2. - 2. * fi) * (2. * tita / pi - 1.)

    aux_normal = np.cross(right_shoulder_direction, aux_vector)
    right_elbow_direction_max_rotation = np.cos(angle) * aux_vector + np.sin(angle) * aux_normal
    
    return np.cross(right_shoulder_direction, right_elbow_direction_max_rotation)


def pose2arm_position(half_pose_landmarks, frame_width, frame_height):
    """
    Given an array of shape (14, 3) or (14, 4) that describes the pose landmarks of the trunk
    of a frame, and the width and height of the frame, returns information of the arms.

    Inputs:
      - half_pose_landmarks (ndarray of shape (14, 3) or (14, 4))
      - frame_width (int)
      - frame_height (int)

    Output: arms_position_dict (dict)
      - left_shoulder_direction (respect to shoulder basis)
      - left_upperarm_length
      - left_shoulder_rotation [0, pi] (respect to max_left_shoulder_rotation)
      - left_elbow_angle
      - left_forearm_length
      - left_elbow_rotation [-pi/2, pi/2]
      - left_wrist_rotation [0, 2pi) (respect to hand basis)
      - left_wrist_inclination [0, pi] (respect to hand basis)
      - left_pinky_direction (respect to hand basis)
      - left_pinky_length
      - left_index_direction (respect to hand basis)
      - left_index_length
      - left_thumb_direction (respect to hand basis)
      - left_thumb_length
      
      - right_shoulder_direction (respect to shoulder basis)
      - right_upperarm_length
      - right_shoulder_rotation [0, pi] (respect to max_right_shoulder_rotation)
      - right_elbow_angle
      - right_forearm_length
      - right_elbow_rotation [-pi/2, pi/2]
      - right_wrist_rotation [0, 2pi) (respect to hand basis)
      - right_wrist_inclination [0, pi] (respect to hand basis)
      - right_pinky_direction (respect to hand basis)
      - right_pinky_length
      - right_index_direction (respect to hand basis)
      - right_index_length
      - right_thumb_direction (respect to hand basis)
      - right_thumb_length
    """
    scale = np.array([frame_width, frame_height, frame_width], dtype=np.float64)
    half_pose_landmarks_ = half_pose_landmarks[:, :3] * scale


    left_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    left_shoulder_basis[:, 0] = half_pose_landmarks_[0] - half_pose_landmarks_[1]
    left_shoulder_basis[:, 2] = np.cross(left_shoulder_basis[:, 0], half_pose_landmarks_[12] - half_pose_landmarks_[0])
    left_shoulder_basis[:, 1] = np.cross(left_shoulder_basis[:, 2], left_shoulder_basis[:, 0])
    left_shoulder_basis /= np.linalg.norm(left_shoulder_basis, axis=0, keepdims=True)

    right_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    right_shoulder_basis[:, 0] = half_pose_landmarks_[1] - half_pose_landmarks_[0]
    right_shoulder_basis[:, 2] = np.cross(right_shoulder_basis[:, 0], half_pose_landmarks_[13] - half_pose_landmarks_[1])
    right_shoulder_basis[:, 1] = np.cross(right_shoulder_basis[:, 2], right_shoulder_basis[:, 0])
    right_shoulder_basis /= np.linalg.norm(right_shoulder_basis, axis=0, keepdims=True)

    left_hand_basis = np.empty((3, 3), dtype=np.float64)
    left_hand_basis[:, 0] = np.cross(half_pose_landmarks_[6] - half_pose_landmarks_[4], half_pose_landmarks_[8] - half_pose_landmarks_[4])
    left_hand_basis[:, 2] = (half_pose_landmarks_[6] + half_pose_landmarks_[8])/2. - half_pose_landmarks_[4]
    left_hand_basis[:, 1] = np.cross(left_hand_basis[:, 2], left_hand_basis[:, 0])
    left_hand_basis /= np.linalg.norm(left_hand_basis, axis=0, keepdims=True)

    right_hand_basis = np.empty((3, 3), dtype=np.float64)
    right_hand_basis[:, 0] = np.cross(half_pose_landmarks_[9] - half_pose_landmarks_[5], half_pose_landmarks_[7] - half_pose_landmarks_[5])
    right_hand_basis[:, 2] = (half_pose_landmarks_[7] + half_pose_landmarks_[9])/2. - half_pose_landmarks_[5]
    right_hand_basis[:, 1] = np.cross(right_hand_basis[:, 2], right_hand_basis[:, 0])
    right_hand_basis /= np.linalg.norm(right_hand_basis, axis=0, keepdims=True)


    left_upperarm_vector = half_pose_landmarks_[2] - half_pose_landmarks_[0]
    left_upperarm_length = np.linalg.norm(left_upperarm_vector)
    left_shoulder_direction = (left_upperarm_vector / left_upperarm_length) @ left_shoulder_basis

    right_upperarm_vector = half_pose_landmarks_[3] - half_pose_landmarks_[1]
    right_upperarm_length = np.linalg.norm(right_upperarm_vector)
    right_shoulder_direction = (right_upperarm_vector / right_upperarm_length) @ right_shoulder_basis

    left_forearm_vector = half_pose_landmarks_[4] - half_pose_landmarks_[2]
    left_forearm_length = np.linalg.norm(left_forearm_vector)
    left_forearm_direction = left_forearm_vector / left_forearm_length
    left_elbow_angle = pi - np.arccos(np.clip(np.dot(left_upperarm_vector, left_forearm_direction) / left_upperarm_length, -1, 1))

    right_forearm_vector = half_pose_landmarks_[5] - half_pose_landmarks_[3]
    right_forearm_length = np.linalg.norm(right_forearm_vector)
    right_forearm_direction = right_forearm_vector / right_forearm_length
    right_elbow_angle = pi - np.arccos(np.clip(np.dot(right_upperarm_vector, right_forearm_direction) / right_upperarm_length, -1, 1))

    left_pinky_vector = (half_pose_landmarks_[6] - half_pose_landmarks_[4]) @ left_hand_basis
    left_pinky_length = np.linalg.norm(left_pinky_vector)
    left_pinky_direction = left_pinky_vector / left_pinky_length
    left_index_vector = (half_pose_landmarks_[8] - half_pose_landmarks_[4]) @ left_hand_basis
    left_index_length = np.linalg.norm(left_index_vector)
    left_index_direction = left_index_vector / left_index_length
    left_thumb_vector = (half_pose_landmarks_[10] - half_pose_landmarks_[4]) @ left_hand_basis
    left_thumb_length = np.linalg.norm(left_thumb_vector)
    left_thumb_direction = left_thumb_vector / left_thumb_length

    right_pinky_vector = (half_pose_landmarks_[7] - half_pose_landmarks_[5]) @ right_hand_basis
    right_pinky_length = np.linalg.norm(right_pinky_vector)
    right_pinky_direction = right_pinky_vector / right_pinky_length
    right_index_vector = (half_pose_landmarks_[9] - half_pose_landmarks_[5]) @ right_hand_basis
    right_index_length = np.linalg.norm(right_index_vector)
    right_index_direction = right_index_vector / right_index_length
    right_thumb_vector = (half_pose_landmarks_[11] - half_pose_landmarks_[5]) @ right_hand_basis
    right_thumb_length = np.linalg.norm(right_thumb_vector)
    right_thumb_direction = right_thumb_vector / right_thumb_length

    left_wrist_direction = - left_forearm_direction @ left_hand_basis
    left_wrist_rotation = azimuth(*(left_wrist_direction * np.array([1, 1, -1], dtype=np.float64)))
    left_wrist_inclination = colatitude(*(left_wrist_direction * np.array([1, 1, -1], dtype=np.float64)))

    right_wrist_direction = - right_forearm_direction @ right_hand_basis
    right_wrist_rotation = azimuth(*(right_wrist_direction * np.array([1, 1, -1], dtype=np.float64)))
    right_wrist_inclination = colatitude(*(right_wrist_direction * np.array([1, 1, -1], dtype=np.float64)))

    a = np.cos(left_wrist_inclination)
    b = np.sin(left_wrist_inclination)
    c = np.cos(left_wrist_rotation)
    d = np.sin(left_wrist_rotation)
    left_palm_normal_vector_no_wrist_inclination = np.array([1.+(a-1.)*c*c, (a-1.)*c*d, b*c], dtype=np.float64)

    a = np.cos(right_wrist_inclination)
    b = np.sin(right_wrist_inclination)
    c = np.cos(right_wrist_rotation)
    d = np.sin(right_wrist_rotation)
    right_palm_normal_vector_no_wrist_inclination = np.array([1.+(a-1.)*c*c, (a-1.)*c*d, b*c], dtype=np.float64)

    if np.isclose(left_elbow_angle, pi, atol=1e-3):
        left_elbow_angle = pi        
        aux_vector = max_left_shoulder_rotation(left_shoulder_direction) @ left_shoulder_basis.T @ left_hand_basis
        left_palm_normal_vector_max_arm_rotation = np.cross(left_wrist_direction, aux_vector)
        if np.dot(left_palm_normal_vector_no_wrist_inclination, aux_vector)>=0:
            left_arm_rotation = np.arccos(np.clip(
                np.dot(left_palm_normal_vector_no_wrist_inclination, left_palm_normal_vector_max_arm_rotation),
                -1, 1))
        else:
            left_arm_rotation = 2.*pi - np.arccos(np.clip(
                np.dot(left_palm_normal_vector_no_wrist_inclination, left_palm_normal_vector_max_arm_rotation),
                -1, 1))
        left_shoulder_rotation = left_arm_rotation / 2.
        left_elbow_rotation = left_arm_rotation - left_shoulder_rotation - pi/2.
    else:
        left_elbow_direction = left_forearm_direction @ left_shoulder_basis
        left_arm_plane_normal_vector = np.cross(left_shoulder_direction, left_elbow_direction)
        left_arm_plane_normal_vector /= np.linalg.norm(left_arm_plane_normal_vector)
        left_arm_max_rotation_plane_normal_vector = max_left_shoulder_rotation(left_shoulder_direction)
        aux_vector = np.cross(left_shoulder_direction, left_arm_max_rotation_plane_normal_vector)
        left_shoulder_rotation = (np.sign(np.dot(left_arm_plane_normal_vector, aux_vector)) *
                                  np.arccos(np.clip(np.dot(left_arm_plane_normal_vector, left_arm_max_rotation_plane_normal_vector), -1, 1)))

        left_palm_normal_vector_no_elbow_rotation = left_arm_plane_normal_vector @ left_shoulder_basis.T @ left_hand_basis
        aux_vector = np.cross(left_palm_normal_vector_no_elbow_rotation, left_wrist_direction)
        left_elbow_rotation = np.sign(np.dot(left_palm_normal_vector_no_wrist_inclination, aux_vector)) * np.arccos(np.clip(
            np.dot(left_palm_normal_vector_no_wrist_inclination, left_palm_normal_vector_no_elbow_rotation), -1, 1))

    if np.isclose(right_elbow_angle, pi, atol=1e-3):
        right_elbow_angle = pi
        aux_vector = max_right_shoulder_rotation(right_shoulder_direction) @ right_shoulder_basis.T @ right_hand_basis
        right_palm_normal_vector_max_arm_rotation = np.cross(right_wrist_direction, aux_vector)
        if np.dot(right_palm_normal_vector_no_wrist_inclination, aux_vector)<=0:
            right_arm_rotation = np.arccos(np.clip(
                np.dot(right_palm_normal_vector_no_wrist_inclination, right_palm_normal_vector_max_arm_rotation),
                -1, 1))
        else:
            right_arm_rotation = 2.*pi - np.arccos(np.clip(
                np.dot(right_palm_normal_vector_no_wrist_inclination, right_palm_normal_vector_max_arm_rotation),
                -1, 1))
        right_shoulder_rotation = right_arm_rotation / 2.
        right_elbow_rotation = right_arm_rotation - right_shoulder_rotation - pi/2.
    else:
        right_elbow_direction = right_forearm_direction @ right_shoulder_basis
        right_arm_plane_normal_vector = np.cross(right_shoulder_direction, right_elbow_direction)
        right_arm_plane_normal_vector /= np.linalg.norm(right_arm_plane_normal_vector)
        right_arm_max_rotation_plane_normal_vector = max_right_shoulder_rotation(right_shoulder_direction)
        aux_vector = np.cross(right_arm_max_rotation_plane_normal_vector, right_shoulder_direction)
        right_shoulder_rotation = (np.sign(np.dot(right_arm_plane_normal_vector, aux_vector)) *
                                   np.arccos(np.clip(np.dot(right_arm_plane_normal_vector, right_arm_max_rotation_plane_normal_vector), -1, 1)))

        right_palm_normal_vector_no_elbow_rotation = - right_arm_plane_normal_vector @ right_shoulder_basis.T @ right_hand_basis
        aux_vector = np.cross(right_palm_normal_vector_no_elbow_rotation, right_wrist_direction)
        right_elbow_rotation = - np.sign(np.dot(right_palm_normal_vector_no_wrist_inclination, aux_vector)) * np.arccos(np.clip(
            np.dot(right_palm_normal_vector_no_wrist_inclination, right_palm_normal_vector_no_elbow_rotation), -1, 1))


    arms_position_dict = {
        "left_shoulder_direction": left_shoulder_direction,
        "left_upperarm_length": left_upperarm_length,
        "left_shoulder_rotation": left_shoulder_rotation,
        "left_elbow_angle": left_elbow_angle,
        "left_forearm_length": left_forearm_length,
        "left_elbow_rotation": left_elbow_rotation,
        "left_wrist_rotation": left_wrist_rotation,
        "left_wrist_inclination": left_wrist_inclination,
        "left_pinky_direction": left_pinky_direction,
        "left_pinky_length": left_pinky_length,
        "left_index_direction": left_index_direction,
        "left_index_length": left_index_length,
        "left_thumb_direction": left_thumb_direction,
        "left_thumb_length": left_thumb_length,

        "right_shoulder_direction": right_shoulder_direction,
        "right_upperarm_length": right_upperarm_length,
        "right_shoulder_rotation": right_shoulder_rotation,
        "right_elbow_angle": right_elbow_angle,
        "right_forearm_length": right_forearm_length,
        "right_elbow_rotation": right_elbow_rotation,
        "right_wrist_rotation": right_wrist_rotation,
        "right_wrist_inclination": right_wrist_inclination,
        "right_pinky_direction": right_pinky_direction,
        "right_pinky_length": right_pinky_length,
        "right_index_direction": right_index_direction,
        "right_index_length": right_index_length,
        "right_thumb_direction": right_thumb_direction,
        "right_thumb_length": right_thumb_length
    }
    return arms_position_dict

def arm_position2pose(arms_position_dict, trunk_landmarks, frame_width, frame_height):
    """
    Given a dictionary like the returned by `pose2arm_position`, an array of shape (4, 3) with
    the landmarks of the trunk and the width and height of the frame, returns an array of shape
    (14, 3) with the landmarks of both arms and the trunk in the following order:
      0 - left shoulder
      1 - right shoulder
      2 - left elbow
      3 - right elbow
      4 - left wrist
      5 - right wrist
      6 - left pinky
      7 - right pinky
      8 - left index
      9 - right index
      10 - left thumb
      11 - right thumb
      12 - left hip
      13 - right hip

    The order of the input landmarks must be:
      0 - left shoulder
      1 - right shoulder
      2 - left hip
      3 - right hip

    Inputs:
      - arms_position_dict (dict)
      - trunk_landmarks (array)
      - frame_width (int)
      - frame_height (int)

    Output:
      - half_pose_landmarks (array)
    """
    scale = np.array([frame_width, frame_height, frame_width], dtype=np.float64)
    half_pose_landmarks = np.empty((14, 3), dtype=np.float64)
    half_pose_landmarks[[0, 1, 12, 13]] = trunk_landmarks * scale


    left_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    left_shoulder_basis[:, 0] = half_pose_landmarks[0] - half_pose_landmarks[1]
    left_shoulder_basis[:, 2] = np.cross(left_shoulder_basis[:, 0], half_pose_landmarks[12] - half_pose_landmarks[0])
    left_shoulder_basis[:, 1] = np.cross(left_shoulder_basis[:, 2], left_shoulder_basis[:, 0])
    left_shoulder_basis /= np.linalg.norm(left_shoulder_basis, axis=0, keepdims=True)

    right_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    right_shoulder_basis[:, 0] = half_pose_landmarks[1] - half_pose_landmarks[0]
    right_shoulder_basis[:, 2] = np.cross(right_shoulder_basis[:, 0], half_pose_landmarks[13] - half_pose_landmarks[1])
    right_shoulder_basis[:, 1] = np.cross(right_shoulder_basis[:, 2], right_shoulder_basis[:, 0])
    right_shoulder_basis /= np.linalg.norm(right_shoulder_basis, axis=0, keepdims=True)


    left_shoulder_direction = arms_position_dict["left_shoulder_direction"] @ left_shoulder_basis.T
    half_pose_landmarks[2] = arms_position_dict["left_upperarm_length"] * left_shoulder_direction + half_pose_landmarks[0]

    right_shoulder_direction = arms_position_dict["right_shoulder_direction"] @ right_shoulder_basis.T
    half_pose_landmarks[3] = arms_position_dict["right_upperarm_length"] * right_shoulder_direction + half_pose_landmarks[1]

    if np.isclose(arms_position_dict["left_elbow_angle"], pi):
        left_forearm_direction = left_shoulder_direction
        left_arm_rotation = arms_position_dict["left_shoulder_rotation"] + arms_position_dict["left_elbow_rotation"] + pi/2.

        aux_vector = max_left_shoulder_rotation(arms_position_dict["left_shoulder_direction"]) @ left_shoulder_basis.T
        left_palm_normal_vector_max_arm_rotation = np.cross(aux_vector, left_forearm_direction)

        left_palm_normal_vector_no_wrist_inclination = (np.cos(left_arm_rotation) * left_palm_normal_vector_max_arm_rotation +
                                                        np.sin(left_arm_rotation) * aux_vector)
    else:
        left_arm_max_rotation_plane_normal_vector = max_left_shoulder_rotation(arms_position_dict["left_shoulder_direction"])
        aux_vector = np.cross(arms_position_dict["left_shoulder_direction"], left_arm_max_rotation_plane_normal_vector)
        left_arm_plane_normal_vector = (np.cos(arms_position_dict["left_shoulder_rotation"]) * left_arm_max_rotation_plane_normal_vector +
                                        np.sin(arms_position_dict["left_shoulder_rotation"]) * aux_vector)

        aux_vector = np.cross(left_arm_plane_normal_vector, arms_position_dict["left_shoulder_direction"])
        left_forearm_direction = (- np.cos(arms_position_dict["left_elbow_angle"]) * arms_position_dict["left_shoulder_direction"] +
                                  np.sin(arms_position_dict["left_elbow_angle"]) * aux_vector) @ left_shoulder_basis.T

        left_palm_normal_vector_no_elbow_rotation = left_arm_plane_normal_vector @ left_shoulder_basis.T
        aux_vector = np.cross(left_forearm_direction, left_palm_normal_vector_no_elbow_rotation)

        left_palm_normal_vector_no_wrist_inclination = (np.cos(arms_position_dict["left_elbow_rotation"]) * left_palm_normal_vector_no_elbow_rotation +
                                                       np.sin(arms_position_dict["left_elbow_rotation"]) * aux_vector)

    if np.isclose(arms_position_dict["right_elbow_angle"], pi):
        right_forearm_direction = right_shoulder_direction
        right_arm_rotation = arms_position_dict["right_shoulder_rotation"] + arms_position_dict["right_elbow_rotation"] + pi/2.

        aux_vector = max_right_shoulder_rotation(arms_position_dict["right_shoulder_direction"]) @ right_shoulder_basis.T
        right_palm_normal_vector_max_arm_rotation = np.cross(aux_vector, right_forearm_direction)

        right_palm_normal_vector_no_wrist_inclination = (np.cos(right_arm_rotation) * right_palm_normal_vector_max_arm_rotation -
                                                       np.sin(right_arm_rotation) * aux_vector)
    else:
        right_arm_max_rotation_plane_normal_vector = max_right_shoulder_rotation(arms_position_dict["right_shoulder_direction"])
        aux_vector = np.cross(right_arm_max_rotation_plane_normal_vector, arms_position_dict["right_shoulder_direction"])
        right_arm_plane_normal_vector = (np.cos(arms_position_dict["right_shoulder_rotation"]) * right_arm_max_rotation_plane_normal_vector +
                                         np.sin(arms_position_dict["right_shoulder_rotation"]) * aux_vector)

        aux_vector = np.cross(right_arm_plane_normal_vector, arms_position_dict["right_shoulder_direction"])
        right_forearm_direction = (- np.cos(arms_position_dict["right_elbow_angle"]) * arms_position_dict["right_shoulder_direction"] +
                                   np.sin(arms_position_dict["right_elbow_angle"]) * aux_vector) @ right_shoulder_basis.T

        right_palm_normal_vector_no_elbow_rotation = - right_arm_plane_normal_vector @ right_shoulder_basis.T
        aux_vector = np.cross(right_forearm_direction, right_palm_normal_vector_no_elbow_rotation)

        right_palm_normal_vector_no_wrist_inclination = (np.cos(arms_position_dict["right_elbow_rotation"]) * right_palm_normal_vector_no_elbow_rotation -
                                                       np.sin(arms_position_dict["right_elbow_rotation"]) * aux_vector)        

    half_pose_landmarks[4] = arms_position_dict["left_forearm_length"] * left_forearm_direction + half_pose_landmarks[2]
    half_pose_landmarks[5] = arms_position_dict["right_forearm_length"] * right_forearm_direction + half_pose_landmarks[3]

    v1 = left_palm_normal_vector_no_wrist_inclination
    v3 = left_forearm_direction
    v2 = np.cross(v3, v1)

    a = np.cos(arms_position_dict["left_wrist_inclination"])
    b = np.sin(arms_position_dict["left_wrist_inclination"])
    c = np.cos(arms_position_dict["left_wrist_rotation"])
    d = np.sin(arms_position_dict["left_wrist_rotation"])
    left_hand_basis = np.empty((3, 3), dtype=np.float64)
    left_hand_basis[:, 0] = (1.+(a-1.)*c*c) * v1 + (a-1.)*c*d * v2 - b*c * v3
    left_hand_basis[:, 1] = (1.+(a-1.)*d*d) * v2 + (a-1.)*c*d * v1 - b*d * v3
    left_hand_basis[:, 2] = b*c * v1 + b*d * v2 + a * v3

    v1 = right_palm_normal_vector_no_wrist_inclination
    v3 = right_forearm_direction
    v2 = np.cross(v3, v1)

    a = np.cos(arms_position_dict["right_wrist_inclination"])
    b = np.sin(arms_position_dict["right_wrist_inclination"])
    c = np.cos(arms_position_dict["right_wrist_rotation"])
    d = np.sin(arms_position_dict["right_wrist_rotation"])
    right_hand_basis = np.empty((3, 3), dtype=np.float64)
    right_hand_basis[:, 0] = (1.+(a-1.)*c*c) * v1 + (a-1.)*c*d * v2 - b*c * v3
    right_hand_basis[:, 1] = (1.+(a-1.)*d*d) * v2 + (a-1.)*c*d * v1 - b*d * v3
    right_hand_basis[:, 2] = b*c * v1 + b*d * v2 + a * v3

    half_pose_landmarks[6] = arms_position_dict["left_pinky_length"] * arms_position_dict["left_pinky_direction"] @ left_hand_basis.T + half_pose_landmarks[4]
    half_pose_landmarks[8] = arms_position_dict["left_index_length"] * arms_position_dict["left_index_direction"] @ left_hand_basis.T + half_pose_landmarks[4]
    half_pose_landmarks[10] = arms_position_dict["left_thumb_length"] * arms_position_dict["left_thumb_direction"] @ left_hand_basis.T + half_pose_landmarks[4]

    half_pose_landmarks[7] = arms_position_dict["right_pinky_length"] * arms_position_dict["right_pinky_direction"] @ right_hand_basis.T + half_pose_landmarks[5]
    half_pose_landmarks[9] = arms_position_dict["right_index_length"] * arms_position_dict["right_index_direction"] @ right_hand_basis.T + half_pose_landmarks[5]
    half_pose_landmarks[11] = arms_position_dict["right_thumb_length"] * arms_position_dict["right_thumb_direction"] @ right_hand_basis.T + half_pose_landmarks[5]

    return half_pose_landmarks / scale


def move_arms(half_pose_landmarks_0, half_pose_landmarks_1, N, frame_width, frame_height):
    """
    Given two arrays of shape (14, 3) representing two "half pose landmarks", an integer N
    and the frame width and height, returns an array of shape (N+1, 14, 3), where each row
    represents a "half pose landmark", and it represents the movement between the two input
    arrays.

    Inputs:
      - half_pose_landmarks_0 (array of shape (14, 3)): initial position
      - half_pose_landmarks_1 (array of shape (14, 3)): final position
      - N (int): number of steps between initial and final positions
      - frame_width (int): width of the frame
      - frame_height (int): height of the frame

    Output: (array of shape (N+1, 14, 3))
    """
    arms_position_0 = pose2arm_position(half_pose_landmarks_0, frame_width, frame_height)
    arms_position_1 = pose2arm_position(half_pose_landmarks_1, frame_width, frame_height)

    def lin_mov(t, a, b):
        return a + t*(b-a)

    lsd_angle = np.arccos(np.clip(np.dot(arms_position_0["left_shoulder_direction"], arms_position_1["left_shoulder_direction"]), -1, 1))
    if np.isclose(lsd_angle, 0):
        def lsd_geo(t):
            return arms_position_0["left_shoulder_direction"]
    elif np.isclose(lsd_angle, pi):
        v = arms_position_0["left_shoulder_direction"][1] * arms_position_0["left_shoulder_direction"]
        v[1] -= 1
        v /= np.sqrt(-v[1])
        def lsd_geo(t):
            return np.cos(pi * t) * arms_position_0["left_shoulder_direction"] + np.sin(pi * t) * v
    else:
        lsd_cte1 = 1. / np.sin(lsd_angle)
        lsd_cte2 = lsd_cte1 * np.cos(lsd_angle)
        az_0 = azimuth(*(arms_position_0["left_shoulder_direction"][[0, 2, 1]] * np.array([-1, 1, 1])))
        az_1 = azimuth(*(arms_position_1["left_shoulder_direction"][[0, 2, 1]] * np.array([-1, 1, 1])))
        if ((arms_position_0["left_shoulder_direction"][2] * arms_position_1["left_shoulder_direction"][2] < 0) and
            (az_1 > az_0 + pi) or (az_1 < az_0 - pi)):
            def lsd_geo(t):
                return ((np.cos((2.*pi-lsd_angle)*t) + np.sin((2.*pi-lsd_angle)*t)*lsd_cte2) * arms_position_0["left_shoulder_direction"] -
                       np.sin((2.*pi-lsd_angle)*t)*lsd_cte1 * arms_position_1["left_shoulder_direction"])
        else:
            def lsd_geo(t):
                return ((np.cos(lsd_angle*t) - np.sin(lsd_angle*t)*lsd_cte2) * arms_position_0["left_shoulder_direction"] +
                        np.sin(lsd_angle*t)*lsd_cte1 * arms_position_1["left_shoulder_direction"])

    rsd_angle = np.arccos(np.clip(np.dot(arms_position_0["right_shoulder_direction"], arms_position_1["right_shoulder_direction"]), -1, 1))
    if np.isclose(rsd_angle, 0):
        def rsd_geo(t):
            return arms_position_0["right_shoulder_direction"]
    elif np.isclose(lsd_angle, pi):
        v = arms_position_0["right_shoulder_direction"][1] * arms_position_0["right_shoulder_direction"]
        v[1] -= 1
        v /= np.sqrt(-v[1])
        def rsd_geo(t):
            return np.cos(pi * t) * arms_position_0["right_shoulder_direction"] + np.sin(pi * t) * v
    else:
        rsd_cte1 = 1. / np.sin(rsd_angle)
        rsd_cte2 = rsd_cte1 * np.cos(rsd_angle)
        az_0 = azimuth(*(arms_position_0["right_shoulder_direction"][[0, 2, 1]] * np.array([-1, 1, 1])))
        az_1 = azimuth(*(arms_position_1["right_shoulder_direction"][[0, 2, 1]] * np.array([-1, 1, 1])))
        if ((arms_position_0["right_shoulder_direction"][2] * arms_position_1["right_shoulder_direction"][2] < 0) and
           (az_1 > az_0 + np.float64(pi)) or (az_1 < az_0 - np.float64(pi))):
            def rsd_geo(t):
                return ((np.cos((2.*pi-rsd_angle)*t) + np.sin((2.*pi-rsd_angle)*t)*rsd_cte2) * arms_position_0["right_shoulder_direction"] -
                       np.sin((2.*pi-rsd_angle)*t)*rsd_cte1 * arms_position_1["right_shoulder_direction"])
        else:
            def rsd_geo(t):
                return ((np.cos(rsd_angle*t) - np.sin(rsd_angle*t)*rsd_cte2) * arms_position_0["right_shoulder_direction"] +
                        np.sin(rsd_angle*t)*rsd_cte1 * arms_position_1["right_shoulder_direction"])

    if np.isclose(arms_position_0["left_wrist_rotation"], arms_position_1["left_wrist_rotation"], atol=1e-2):
        def lw_mov(t):
            return (lin_mov(t, arms_position_0["left_wrist_rotation"], arms_position_1["left_wrist_rotation"]),
                   lin_mov(t, arms_position_0["left_wrist_inclination"], arms_position_1["left_wrist_inclination"]))
    else:
        total_desp = arms_position_0["left_wrist_inclination"] + arms_position_1["left_wrist_inclination"]
        def lw_mov(t):
            if t * total_desp < arms_position_0["left_wrist_inclination"]:
                return arms_position_0["left_wrist_rotation"], arms_position_0["left_wrist_inclination"] - total_desp * t
            else:
                return arms_position_1["left_wrist_rotation"], - arms_position_0["left_wrist_inclination"] + total_desp * t

    if np.isclose(arms_position_0["right_wrist_rotation"], arms_position_1["right_wrist_rotation"], atol=1e-2):
        def rw_mov(t):
            return (lin_mov(t, arms_position_0["right_wrist_rotation"], arms_position_1["right_wrist_rotation"]),
                   lin_mov(t, arms_position_0["right_wrist_inclination"], arms_position_1["right_wrist_inclination"]))
    else:
        total_desp = arms_position_0["right_wrist_inclination"] + arms_position_1["right_wrist_inclination"]
        def rw_mov(t):
            if t * total_desp < arms_position_0["right_wrist_inclination"]:
                return arms_position_0["right_wrist_rotation"], arms_position_0["right_wrist_inclination"] - total_desp * t
            else:
                return arms_position_1["right_wrist_rotation"], - arms_position_0["right_wrist_inclination"] + total_desp * t

    left_thumb_direction_0 = arms_position_0["left_thumb_direction"]
    left_thumb_direction_1 = arms_position_1["left_thumb_direction"]
    ltd_angle = np.arccos(np.clip(np.dot(left_thumb_direction_0, left_thumb_direction_1), -1, 1))
    if np.isclose(ltd_angle, 0):
        def ltd_geo(t):
            return arms_position_0["left_thumb_direction"]
    else:
        ltd_cte1 = 1. / np.sin(ltd_angle)
        ltd_cte2 = ltd_cte1 * np.cos(ltd_angle)
        def ltd_geo(t):
            return (np.cos(ltd_angle*t) - np.sin(ltd_angle*t)*ltd_cte2) * left_thumb_direction_0 + np.sin(ltd_angle*t)*ltd_cte1 * left_thumb_direction_1

    right_thumb_direction_0 = arms_position_0["right_thumb_direction"]
    right_thumb_direction_1 = arms_position_1["right_thumb_direction"]
    rtd_angle = np.arccos(np.clip(np.dot(right_thumb_direction_0, right_thumb_direction_1), -1, 1))
    if np.isclose(rtd_angle, 0):
        def rtd_geo(t):
            return arms_position_0["right_thumb_direction"]
    else:
        rtd_cte1 = 1. / np.sin(rtd_angle)
        rtd_cte2 = rtd_cte1 * np.cos(rtd_angle)
        def rtd_geo(t):
            return (np.cos(rtd_angle*t) - np.sin(rtd_angle*t)*rtd_cte2) * right_thumb_direction_0 + np.sin(rtd_angle*t)*rtd_cte1 * right_thumb_direction_1

    trunk_landmarks = half_pose_landmarks_0[[0, 1, 12, 13]]
    arms_position_var = arms_position_0.copy()

    def movement_progress(t):
        return (3. - 2.*t)*t*t

    half_pose_landmark_movement = np.empty((N+1, 14, 3), dtype=np.float64)
    half_pose_landmark_movement[0] = arm_position2pose(arms_position_var, trunk_landmarks, frame_width, frame_height)

    for i in range(1, N+1):
        t = i/N
        s = movement_progress(t)

        arms_position_var["left_shoulder_direction"] = lsd_geo(s)
        arms_position_var["left_shoulder_rotation"] = lin_mov(s, arms_position_0["left_shoulder_rotation"], arms_position_1["left_shoulder_rotation"])
        arms_position_var["left_elbow_angle"] = lin_mov(s, arms_position_0["left_elbow_angle"], arms_position_1["left_elbow_angle"])
        arms_position_var["left_elbow_rotation"] = lin_mov(s, arms_position_0["left_elbow_rotation"], arms_position_1["left_elbow_rotation"])
        arms_position_var["left_wrist_rotation"], arms_position_var["left_wrist_inclination"] = lw_mov(s)
        arms_position_var["left_thumb_direction"] = ltd_geo(s)

        arms_position_var["right_shoulder_direction"] = rsd_geo(s)
        arms_position_var["right_shoulder_rotation"] = lin_mov(s, arms_position_0["right_shoulder_rotation"], arms_position_1["right_shoulder_rotation"])
        arms_position_var["right_elbow_angle"] = lin_mov(s, arms_position_0["right_elbow_angle"], arms_position_1["right_elbow_angle"])
        arms_position_var["right_elbow_rotation"] = lin_mov(s, arms_position_0["right_elbow_rotation"], arms_position_1["right_elbow_rotation"])
        arms_position_var["right_wrist_rotation"], arms_position_var["right_wrist_inclination"] = rw_mov(s)
        arms_position_var["right_thumb_direction"] = rtd_geo(s)

        half_pose_landmark_movement[i] = arm_position2pose(arms_position_var, trunk_landmarks, frame_width, frame_height)

    return half_pose_landmark_movement
