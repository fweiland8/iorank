import numpy as np
import math


def angle_between_radians(v1, v2):
    v1_u = normalize_vec(v1)
    v2_u = normalize_vec(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_degrees(v1, v2):
    return math.degrees(angle_between_radians(v1, v2))


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec2 - vec1)


def normalize_vec(vector):
    # Although a zero vector can not be normalized, it is returned for robustness reasons
    if np.array_equal(np.zeros_like(vector), vector):
        return vector

    return vector / np.linalg.norm(vector)


def are_linear_independent(vec1, vec2):
    matrix = np.transpose(np.vstack((vec1, vec2)))
    return np.linalg.matrix_rank(matrix) == 2


def get_orthogonal_vec(vec):
    return np.array([- vec[1], vec[0]])
