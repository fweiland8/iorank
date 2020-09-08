import matplotlib.pyplot as plt

from iorank.util.util import is_dummy_mask, is_dummy_box
from iorank.visualization.visualization_utils import draw_torch_image, add_mask, draw_box


def show_image_with_masks(image, masks, boxes=None, scores=None):
    """
    Plots the given image together with the provided object masks. Optionally,
    bounding boxes and utility scores can be plotted.

    :param image: Image to be plotted
    :param masks: Object masks
    :param boxes: Optional. Bounding box coordinates.
    :param scores: Optional. Utility scores from object ranking
    """

    # Clone image because it will be modified
    image = image.clone()

    fig, ax = plt.subplots(1)
    for mask in masks:
        if not is_dummy_mask(mask):
            add_mask(image, mask)

    if boxes is not None and scores is not None:
        for box, score in zip(boxes, scores):
            if not is_dummy_box(box):
                draw_box(ax, box, annotation="{:0.4f}".format(score))
    elif boxes is not None:
        for box in boxes:
            if not is_dummy_box(box):
                draw_box(ax, box)

    draw_torch_image(ax, image)

    plt.show()


def show_image_with_boxes(image, boxes, ranking=None):
    """
    Plots the given image together with the provided bounding boxes. Optionally,
    ranks of the obects can also be plotted.

    :param image: Image to be plotted
    :param boxes:  Bounding box coordinates
    :param ranking: Optional. Ranks from object ranking
    """
    fig, ax = plt.subplots(1)
    if ranking is not None:
        for box, rank in zip(boxes, ranking):
            if not is_dummy_box(box):
                draw_box(ax, box, annotation="{:d}".format(rank))
    else:
        for box in boxes:
            if not is_dummy_box(box):
                draw_box(ax, box)

    draw_torch_image(ax, image)

    plt.show()
