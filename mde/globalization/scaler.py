import cv2 as cv
import numpy as np
from skimage.transform import rescale

from mde.feature_extraction.shapes import RectangularSegment, Segment


def rescale_array(array, scaling_factors):
    return rescale(array, scaling_factors)


def resize_array(array, new_size):
    return cv.resize(array, new_size)


# TODO: Find a better name
# TODO: This can be done beforehand! To reduce computational time
def mask_array_varying_size(array, segment):
    if isinstance(segment, Segment):
        mask = segment.get_mask
    else:
        mask = segment

    if np.array_equal(array.shape, mask.shape):
        return segment.get_mask

    if type(segment) is RectangularSegment:
        h_start = np.ceil(segment.get_start_h / mask.shape[0] * array.shape[0]).astype('int')
        h_end = np.ceil(segment.get_end_h / mask.shape[0] * array.shape[0]).astype('int')
        w_start = np.ceil(segment.get_start_w / mask.shape[1] * array.shape[1]).astype('int')
        w_end = np.ceil(segment.get_end_w / mask.shape[1] * array.shape[1]).astype('int')

        result = np.zeros(array.shape)
        result[h_start: h_end, w_start: w_end] = 1
        return result.astype(bool)

    if mask.ndim == 3:
        mask = np.any(mask, axis=-1)

    # Assuming mask is greater than array
    scaling_factors = np.array(array.shape) / np.array(mask.shape)
    return rescale(mask.astype(float), scaling_factors) > 0.25
