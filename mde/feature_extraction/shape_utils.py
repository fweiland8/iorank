import cv2 as cv
import logging
import numpy as np
import sys

from mde.feature_extraction.shapes import Line
from mde.math_utils import euclidean_distance, normalize_vec, are_linear_independent, angle_between_degrees


def detect_hough_lines(image, canny_th1=None, canny_th2=None, canny_size=3):
    imgray = cv.cvtColor(image.astype('uint8'), cv.COLOR_RGB2GRAY)

    if canny_th1 is None or canny_th2 is None:
        canny_th2, thresh_im = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        canny_th1 = 0.5 * canny_th2

    edges = cv.Canny(imgray, canny_th1, canny_th2, apertureSize=canny_size)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=50, maxLineGap=10)

    # Result is a list of entries consisting of the coordinate x1, y1, x2 and y2 of the line beginning and end

    result = []
    for i in range(len(lines)):
        line = lines[i, 0]  # x1, y1, x2, y2
        result.append(Line.build_from_points(np.array([line[1], line[0]]), np.array([line[3], line[2]])))

    return result


# Lines are of shape (num_lines, 1, 4)
def line_length_filter(lines, threshold):
    result = []
    for i in range(len(lines)):
        if euclidean_distance(lines[i].p2, lines[i].p1) > threshold:
            result.append(lines[i])
    return result


def line_angle_filter(lines, threshold):
    result = []

    for i in range(len(lines)):
        if np.abs(lines[i].p1[0] - lines[i].p2[0]) > threshold and np.array(
                lines[i].p1[1] - lines[i].p2[1]) > threshold:
            result.append(lines[i])
    return result


# TODO: What is a reasonable threshold?
def line_fusion_filter(lines, angle_threshold=5, distance_threshold=10, apply_once=True):
    result = []
    remove_idxs = []

    logging.debug("Processing line fusion filter for {} lines...".format(len(lines)))

    for i in range(len(lines)):
        if i in remove_idxs:
            continue

        for j in range(i + 1, len(lines)):
            if i in remove_idxs or j in remove_idxs:
                continue

            line1 = lines[i]
            line2 = lines[j]

            angle = angle_between_degrees(line1.direction, line2.direction)
            if angle > 90:
                angle = 180 - angle

            line_dist = min_line_distance(line1, line2)

            if angle < angle_threshold and line_dist < distance_threshold:
                # Merge lines, remove lines i and j from lines, add merged line to result
                merged_line = merge_lines(line1, line2)
                result.append(merged_line)
                remove_idxs.append(i)
                remove_idxs.append(j)

    result_len = len(result)

    for i in range(len(lines)):
        if i in remove_idxs:
            continue

        result.append(lines[i])

    if result_len > 0 and not apply_once:
        result = line_fusion_filter(np.array(result), angle_threshold=angle_threshold,
                                    distance_threshold=distance_threshold,
                                    apply_once=apply_once)

    return result


def realign_line(line, point):
    """
    Realigns line s.t. the nearest point to a certain point is the new support vector
    """
    line.normalize_direction()

    dist1 = euclidean_distance(line.p1, point)
    dist2 = euclidean_distance(line.p2, point)

    if dist1 >= dist2:
        res_line = Line.build_from_points(line.p2, line.p1)
    else:
        res_line = line

    res_line.normalize_direction()
    return res_line


def merge_lines(line1, line2):
    def get_weight(line1, line2):
        length1 = euclidean_distance(line1.p1, line1.p2)
        length2 = euclidean_distance(line2.p1, line2.p2)
        return length1 / (length1 + length2)

    weight1 = get_weight(line1, line2)
    weight2 = 1. - weight1

    x = weight1 * ((line1.x1 + line1.x2) / 2.) + weight2 * ((line2.x1 + line2.x2) / 2.)
    y = weight1 * ((line1.y1 + line1.y2) / 2.) + weight2 * ((line2.y1 + line2.y2) / 2.)

    vec1 = line1.direction
    vec2 = line2.direction

    if angle_between_degrees(vec1, vec2) > 90:
        # Invert vector 2
        vec2 *= -1

    if are_linear_independent(vec1, vec2):
        dir = weight1 * vec1 + weight2 * vec2
        dir = normalize_vec(dir)
    else:
        dir = vec1

    assert len(dir) == 2

    supp_line = Line(np.array([x, y]), dir)
    return build_line(supp_line, line1, line2)


def build_line(support_line, line1, line2):
    max_dist = -sys.float_info.max
    min_dist = sys.float_info.max

    for elem in [line1.p1, line1.p2, line2.p1, line2.p2]:
        dist = distance_from_support(support_line, elem)
        max_dist = max(max_dist, dist)
        min_dist = min(min_dist, dist)

    # Apply is just the multiplication of scalar t to the direction added by the support
    p1 = support_line.direction * max_dist + support_line.support_vector
    p2 = support_line.direction * min_dist + support_line.support_vector
    new_dir = p2 - p1
    return Line(p1, new_dir)

    # dir = np.array([support_line[2] - support_line[0], support_line[3] - support_line[1]])
    # vec1 = support_line[:2] + max_dist * dir
    # vec2 = support_line[:2] + min_dist * dir
    #
    # return np.hstack((vec1, vec2))


def distance_from_support(supp_line, point):
    ortho_line = supp_line.get_orthogonal()
    ortho_line.set_support_vector(point)

    intersection = supp_line.intersection(ortho_line)
    if intersection is None:
        logging.warning(
            "No intersection could be determined when calculating the distance from the support line given a point.")

    return supp_line.unapply(intersection)


# Line consists of list of the coordinate x1, y1, x2, y2
def min_line_distance(line1, line2):
    if line1.is_intersecting(line2):
        return 0

    result = pt_line_dist(line1, line2.p1)
    result = min(result, pt_line_dist(line1, line2.p2))
    result = min(result, pt_line_dist(line2, line1.p1))
    result = min(result, pt_line_dist(line2, line1.p2))

    return result


def pt_line_dist(line, point):
    x1, y1 = line.p1
    x2, y2 = line.p2
    px, py = point

    x2 -= x1
    y2 -= y1
    px -= x1
    py -= y1

    dotprod = px * x2 + py * y2
    if dotprod <= 0.:
        proj_len_sq = 0.
    else:
        px = x2 - px
        py = y2 - py
        dotprod = px * x2 + py * y2
        if dotprod <= 0.:
            proj_len_sq = 0.
        else:
            proj_len_sq = dotprod * dotprod / (x2 * x2 + y2 * y2)

    len_sq = px * px + py * py - proj_len_sq
    if len_sq < 0.:
        len_sq = 0.

    return np.sqrt(len_sq)
