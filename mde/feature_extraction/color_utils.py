from matplotlib.colors import rgb_to_hsv
import numpy as np


def convert_image_rgb_to_hsv(image):
    assert len(image.shape) == 3 and image.shape[
        2] == 3, "The image is assumed to have three dimensions and three channels!"

    return rgb_to_hsv(image)


def get_saturation(image):
    """
    Function calculating the saturation component of a HSV / HSL color model given RGB values
    :param image: RGB image
    :return: Returns the saturation for each pixel of the input image
    """
    max_seg_im = np.max(image, axis=-1)
    min_seg_im = np.min(image, axis=-1)
    return np.where(max_seg_im == 0, 0, (max_seg_im - min_seg_im) / (max_seg_im + 1e-15))
