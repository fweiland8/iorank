import torch
from matplotlib import patches


def draw_box(ax, box, color='r', annotation=None):
    """
    Draws the given box to the given axis.

    :param ax: Matplotlib axis
    :param box: Bounding box coordinates of the form (x0,y0,x1,y1)
    :param color: Color with which the boxes are to be plotted. Default: 'r' (red)
    :param annotation: Optional annotation that is drawn next to the box
    """

    start = box[0:2]
    rect = patches.Rectangle(start, box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=color,
                             facecolor='none')
    ax.add_patch(rect)

    if annotation is not None:
        ax.text(box[0], box[1], annotation, color='r', fontsize='x-small')


def draw_torch_image(ax, image_tensor):
    """
    Draws the given image represented as PyTorch tensor to given axis.

    :param ax: Matplotlib axis
    :param image_tensor: Tensor of size (3,H,W)
    """
    ax.imshow(image_tensor.permute(1, 2, 0))


def add_mask(image_tensor, mask, channel=0, alpha=0.5):
    """
    Draws a mask to the given image.

    :param image_tensor: Image tensor of size (3,H,W)
    :param mask: Object mask
    :param channel: Channel in which the object mask is to be drawn. Default: 0 (red)
    :param alpha: Alpha value with which the mask is to be drawn. The higher the more the
    mask is visible. Default: 0.5
    :return: Image to which the mask has been drawn
    """

    masked_channel = image_tensor[channel]
    masked_channel = (1 - alpha) * masked_channel + alpha * torch.ones(image_tensor.size()[-2:])
    c = torch.where(mask == 1, masked_channel, image_tensor[channel])
    image_tensor[channel] = c
    return image_tensor
