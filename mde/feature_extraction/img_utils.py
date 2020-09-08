import numpy as np


def crop_segment(segment, image):
    """
    Function cropping a rectangle covering the segment provided as boolean mask. The reason for possibly extending the
    segment area in case of curvy boundaries is to get a proper crop for feature calculation often requiring rectangles
    as input.

    :param segment: Boolean mask sharing the shape of the image. True entries indicate membership to the segment
    :param image: Original image of the shape [height, width, (channels)]
    :return: Returns a rectangle crop of the image's content given the segment mask
    """
    assert np.array_equal(segment.shape[:2], image.shape[:2]), "The shape of the segment must match the image's shape" \
                                                               "(at least the first two positions)!"

    min_axis0 = -1
    for i in range(segment.shape[0]):
        if segment[i].any():
            min_axis0 = i
            break
    max_axis0 = -1
    for i in range(segment.shape[0]):
        if segment[segment.shape[0] - 1 - i].any():
            max_axis0 = segment.shape[0] - i
            break

    min_axis1 = -1
    for i in range(segment.shape[1]):
        if segment[:, i].any():
            min_axis1 = i
            break
    max_axis1 = -1
    for i in range(segment.shape[1]):
        if segment[:, segment.shape[1] - 1 - i].any():
            max_axis1 = segment.shape[1] - i
            break

    if min_axis0 == -1 or max_axis0 == -1 or min_axis1 == -1 or max_axis1 == -1:
        return None

    return image[min_axis0: max_axis0, min_axis1: max_axis1]
