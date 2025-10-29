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


def landmarks2arm_position(half_landmarks, frame_width, frame_height):
    """
    Given an array of shape (48, 3) that describes the landmarks of the trunk of a
    frame, and the width and height of the frame, returns information of the arms.

    Inputs:
      - half_landmarks (ndarray of shape (48, 3))
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
      - left_finger_direcions (respect to hand basis)
      - left_finger_lengths
      
      - right_shoulder_direction (respect to shoulder basis)
      - right_upperarm_length
      - right_shoulder_rotation [0, pi] (respect to max_right_shoulder_rotation)
      - right_elbow_angle
      - right_forearm_length
      - right_elbow_rotation [-pi/2, pi/2]
      - right_wrist_rotation [0, 2pi) (respect to hand basis)
      - right_wrist_inclination [0, pi] (respect to hand basis)
      - right_finger_direcions (respect to hand basis)
      - right_finger_lengths
    """
    scale = np.array([frame_width, frame_height, frame_width], dtype=np.float64)
    half_landmarks_ = half_landmarks * scale


    left_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    left_shoulder_basis[:, 0] = half_landmarks_[2] - half_landmarks_[3]
    left_shoulder_basis[:, 2] = np.cross(left_shoulder_basis[:, 0], half_landmarks_[0] - half_landmarks_[2])
    left_shoulder_basis[:, 1] = np.cross(left_shoulder_basis[:, 2], left_shoulder_basis[:, 0])
    left_shoulder_basis /= np.linalg.norm(left_shoulder_basis, axis=0, keepdims=True)

    right_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    right_shoulder_basis[:, 0] = half_landmarks_[3] - half_landmarks_[2]
    right_shoulder_basis[:, 2] = np.cross(right_shoulder_basis[:, 0], half_landmarks_[1] - half_landmarks_[3])
    right_shoulder_basis[:, 1] = np.cross(right_shoulder_basis[:, 2], right_shoulder_basis[:, 0])
    right_shoulder_basis /= np.linalg.norm(right_shoulder_basis, axis=0, keepdims=True)

    left_hand_basis = np.empty((3, 3), dtype=np.float64)
    left_hand_basis[:, 0] = np.cross(half_landmarks_[40] - half_landmarks_[6], half_landmarks_[16] - half_landmarks_[6])
    left_hand_basis[:, 2] = (half_landmarks_[40] + half_landmarks_[16])/2. - half_landmarks_[6]
    left_hand_basis[:, 1] = np.cross(left_hand_basis[:, 2], left_hand_basis[:, 0])
    left_hand_basis /= np.linalg.norm(left_hand_basis, axis=0, keepdims=True)

    right_hand_basis = np.empty((3, 3), dtype=np.float64)
    right_hand_basis[:, 0] = np.cross(half_landmarks_[17] - half_landmarks_[7], half_landmarks_[41] - half_landmarks_[7])
    right_hand_basis[:, 2] = (half_landmarks_[41] + half_landmarks_[17])/2. - half_landmarks_[7]
    right_hand_basis[:, 1] = np.cross(right_hand_basis[:, 2], right_hand_basis[:, 0])
    right_hand_basis /= np.linalg.norm(right_hand_basis, axis=0, keepdims=True)


    left_upperarm_vector = half_landmarks_[4] - half_landmarks_[2]
    left_upperarm_length = np.linalg.norm(left_upperarm_vector)
    left_shoulder_direction = (left_upperarm_vector / left_upperarm_length) @ left_shoulder_basis

    right_upperarm_vector = half_landmarks_[5] - half_landmarks_[3]
    right_upperarm_length = np.linalg.norm(right_upperarm_vector)
    right_shoulder_direction = (right_upperarm_vector / right_upperarm_length) @ right_shoulder_basis

    left_forearm_vector = half_landmarks_[6] - half_landmarks_[4]
    left_forearm_length = np.linalg.norm(left_forearm_vector)
    left_forearm_direction = left_forearm_vector / left_forearm_length
    left_elbow_angle = pi - np.arccos(np.clip(np.dot(left_upperarm_vector, left_forearm_direction) / left_upperarm_length, -1, 1))

    right_forearm_vector = half_landmarks_[7] - half_landmarks_[5]
    right_forearm_length = np.linalg.norm(right_forearm_vector)
    right_forearm_direction = right_forearm_vector / right_forearm_length
    right_elbow_angle = pi - np.arccos(np.clip(np.dot(right_upperarm_vector, right_forearm_direction) / right_upperarm_length, -1, 1))


    left_finger_directions = np.empty((4, 5, 3), dtype=np.float64)
    right_finger_directions = np.empty((4, 5, 3), dtype=np.float64)
    left_finger_directions[0] = half_landmarks_[8::8] - half_landmarks_[6]
    right_finger_directions[0] = half_landmarks_[9::8] - half_landmarks_[7]
    for i in range(1, 4):
        left_finger_directions[i] = (half_landmarks_[8+2*i::8] - half_landmarks_[6+2*i::8])
        right_finger_directions[i] = (half_landmarks_[9+2*i::8] - half_landmarks_[7+2*i::8])
    left_finger_directions @= left_hand_basis
    right_finger_directions @= right_hand_basis

    left_finger_lengths = np.linalg.norm(left_finger_directions, axis=2, keepdims=True)
    right_finger_lengths = np.linalg.norm(right_finger_directions, axis=2, keepdims=True)
    left_finger_directions /= left_finger_lengths
    right_finger_directions /= right_finger_lengths


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
        "left_finger_directions": left_finger_directions,
        "left_finger_lengths": left_finger_lengths,

        "right_shoulder_direction": right_shoulder_direction,
        "right_upperarm_length": right_upperarm_length,
        "right_shoulder_rotation": right_shoulder_rotation,
        "right_elbow_angle": right_elbow_angle,
        "right_forearm_length": right_forearm_length,
        "right_elbow_rotation": right_elbow_rotation,
        "right_wrist_rotation": right_wrist_rotation,
        "right_wrist_inclination": right_wrist_inclination,
        "right_finger_directions": right_finger_directions,
        "right_finger_lengths": right_finger_lengths
    }
    return arms_position_dict

def arm_position2landmarks(arms_position_dict, trunk_landmarks, frame_width, frame_height):
    """
    Given a dictionary like the returned by `landmarks2arm_position`, an array of shape (4, 3)
    with the landmarks of the trunk and the width and height of the frame, returns an array of
    shape (48, 3) with the landmarks of both arms and hands and the trunk, in the following order:
      0 - left hip
      1 - right hip
      2 - left shoulder
      3 - right shoulder
      4 - left elbow
      5 - right elbow
      6 - left wrist
      7 - right wrist
      8, 10, 12, 14 - left thumb
      9, 11, 13, 15 - right thumb
      16, 18, 20, 22 - left index
      17, 19, 21, 23 - right index
      24, 26, 28, 30 - left middle
      25, 27, 29, 31 - right middle
      32, 34, 36, 38 - left ring
      33, 35, 37, 39 - right ring
      40, 42, 44, 46 - left pinky
      41, 43, 45, 47 - right pinky

    The order of the input landmarks must be:
      0 - left hip
      1 - right hip
      2 - left shoulder
      3 - right shoulder

    Inputs:
      - arms_position_dict (dict)
      - trunk_landmarks (array)
      - frame_width (int)
      - frame_height (int)

    Output:
      - half_landmarks (array)
    """
    scale = np.array([frame_width, frame_height, frame_width], dtype=np.float64)
    half_landmarks = np.empty((48, 3), dtype=np.float64)
    half_landmarks[:4] = trunk_landmarks * scale


    left_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    left_shoulder_basis[:, 0] = half_landmarks[2] - half_landmarks[3]
    left_shoulder_basis[:, 2] = np.cross(left_shoulder_basis[:, 0], half_landmarks[0] - half_landmarks[2])
    left_shoulder_basis[:, 1] = np.cross(left_shoulder_basis[:, 2], left_shoulder_basis[:, 0])
    left_shoulder_basis /= np.linalg.norm(left_shoulder_basis, axis=0, keepdims=True)

    right_shoulder_basis = np.empty((3, 3), dtype=np.float64)
    right_shoulder_basis[:, 0] = half_landmarks[3] - half_landmarks[2]
    right_shoulder_basis[:, 2] = np.cross(right_shoulder_basis[:, 0], half_landmarks[1] - half_landmarks[3])
    right_shoulder_basis[:, 1] = np.cross(right_shoulder_basis[:, 2], right_shoulder_basis[:, 0])
    right_shoulder_basis /= np.linalg.norm(right_shoulder_basis, axis=0, keepdims=True)


    left_shoulder_direction = arms_position_dict["left_shoulder_direction"] @ left_shoulder_basis.T
    half_landmarks[4] = arms_position_dict["left_upperarm_length"] * left_shoulder_direction + half_landmarks[2]

    right_shoulder_direction = arms_position_dict["right_shoulder_direction"] @ right_shoulder_basis.T
    half_landmarks[5] = arms_position_dict["right_upperarm_length"] * right_shoulder_direction + half_landmarks[3]

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

    half_landmarks[6] = arms_position_dict["left_forearm_length"] * left_forearm_direction + half_landmarks[4]
    half_landmarks[7] = arms_position_dict["right_forearm_length"] * right_forearm_direction + half_landmarks[5]

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
    
    left_finger_directions = arms_position_dict["left_finger_directions"]
    left_finger_lengths = arms_position_dict["left_finger_lengths"]
    right_finger_directions = arms_position_dict["right_finger_directions"]
    right_finger_lengths = arms_position_dict["right_finger_lengths"]

    half_landmarks[8::8] = left_finger_lengths[0] * left_finger_directions[0] @ left_hand_basis.T + half_landmarks[6]
    half_landmarks[9::8] = right_finger_lengths[0] * right_finger_directions[0] @ right_hand_basis.T + half_landmarks[7]

    for i in range(1, 4):
        half_landmarks[8+2*i::8] = left_finger_lengths[i] * left_finger_directions[i] @ left_hand_basis.T + half_landmarks[6+2*i::8]
        half_landmarks[9+2*i::8] = right_finger_lengths[i] * right_finger_directions[i] @ right_hand_basis.T + half_landmarks[7+2*i::8]

    return half_landmarks / scale