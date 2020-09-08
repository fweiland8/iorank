import cv2
import logging
import numpy as np
import socket
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import uuid
import yaml
from PIL import Image
from torch.utils.data import Subset

logger = logging.getLogger("util")


def get_box_size(box):
    """
    Computes the area that is covered by a bounding box

    :param box: Bounding box coordinates
    :return: Size of the covered area
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height


def get_patch_from_image(box, rgb_image):
    """
    Extract a patch described by bounding box coordinates from the given RGB image.

    :param box: Bounding box coordinates in the form (x0,y0,x1,y1)
    :param rgb_image: The RGB image as tensor of size (3,H,W)
    :return: The extracted patch
    """
    # TODO: Test me
    if torch.is_tensor(box):
        x0, y0, x1, y1 = [v.item() for v in box.type(torch.int)]
    else:
        x0, y0, x1, y1 = box
    height = int(y1 - y0) + 1
    width = int(x1 - x0) + 1
    return rgb_image[:, y0:y0 + height, x0:x0 + width]


def scores_to_rankings_torch(scores):
    """
    Turns the given utility scores into a ranking. Suitable for PyTorch tensors.

    N: Batch size
    U: Upper bound for the number of objects (padding size)

    :param scores: Utility scores. Tensor of size (N,U)
    :return: Tensor with rankings of size (N,U)
    """
    # Store length (U)
    length = scores.size()[1]
    rankings = []
    for s in scores:
        # Filter out dummy scores
        s = s[s > -1]
        ordering = torch.argsort(s)
        # Reverse ordering
        ordering = torch.flip(ordering, dims=[0])
        ranking = torch.argsort(ordering)
        # Pad back to the original length
        ranking = pad(ranking, length)
        rankings.append(ranking)
    rankings = torch.stack(rankings)
    return rankings


def scores_to_rankings_with_size_torch(scores, ranking_sizes):
    """
    Turns the given utility scores into a ranking. Suitable for PyTorch tensors.

    This method is needed if it cannot be derived from the scores tensor how many real
    (non-dummy) objects there are for an instance. Hence, the ranking sizes are provided
    to this method as additional argument.

    :param scores: Utility scores. Tensor of size (N,U)
    :param ranking_sizes: Size of the rankings. Tensor of size N
    :return: Tensor with rankings of size (N,U)
    """
    rankings = []
    length = scores.size()[1]
    for s, ranking_size in zip(scores, ranking_sizes):
        # Consider the first 'ranking_size' scores
        s = s[:ranking_size]
        ordering = torch.argsort(s)
        # Reverse ordering
        ordering = torch.flip(ordering, dims=[0])
        ranking = torch.argsort(ordering)
        ranking = pad(ranking, length)
        rankings.append(ranking)
    rankings = torch.stack(rankings)
    return rankings


def is_int(s):
    """
    Checks if the given string represents an integer.

    :param s: Some string
    :return: True, if s can be parsed to an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    """
    Checks if the given string represents a float.

    :param s: Some string
    :return: True, if s can be parsed to a float, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def string_to_kwargs(str):
    """
    Generates a kwargs dict from the given string.

    Example: 'param1=value1,param2=value2' is parsed to {'param1:'value1','param2':'value2'}

    :param str: String with key value pairs
    :return: Parsed dict
    """
    pairs = str.split(',')
    kwargs = {}
    for p in pairs:
        key, value = p.split('=')
        if is_int(value):
            kwargs[key] = int(value)
        elif is_float(value):
            kwargs[key] = float(value)
        elif value == 'true':
            kwargs[key] = True
        elif value == 'false':
            kwargs[key] = False
        else:
            kwargs[key] = value
    return kwargs


def get_hostname():
    """
    Returns the hostname of the server the program is executed on.

    :return: The hostname
    """
    return socket.gethostname()


def get_device():
    """
    Returns the suitable PyTorch device for the server the program is executed on.

    :return: cuda device if CUDA is available, cpu device otherwise
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def setup_logging(filename=None, loglevel='DEBUG'):
    """
    Sets up Python logging.

    :param filename: Optional. Full path to the file the log has to be written to. If no filename is provided,
    log outputs are written to stdout.
    :param loglevel: Loglevel to be used. Default: 'DEBUG'
    :return:
    """
    if filename is None:
        h = logging.StreamHandler()
        logging.basicConfig(handlers=[h], format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)
    else:
        logging.basicConfig(filename=filename, filemode='w', format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    # Change PIL loglevel to reduce debug spam
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)


# Method expects and returns PIL image
def blacken_image(rgb_image, box):
    """
    Blackens the given RGB image except for the area inside the provided bounding box.

    :param rgb_image: The RGB image
    :param box: Bounding box coordinates in the form (x0,y0,x1,y1)
    :return: An image which is blackened except for the area inside the provided bounding box
    """
    # Copy patch
    patch = rgb_image.crop(box.cpu().numpy())

    # Paste patch into a fully black image
    new = Image.new('RGB', rgb_image.size, (0, 0, 0))
    offset = (box[0], box[1])
    new.paste(patch, offset)
    return new


def is_dummy_box(box):
    """
    Returns if the provided box describes a dummy bounding box

    :param box: Bounding box coordinates of the form (x0,y0,x1,y1)
    :return: True, if the bounding box describes a dummy box, False otherwise
    """
    # Dummy box = (-1,-1,-1,-1)
    return torch.all(box.eq(-1))


def is_dummy_mask(mask):
    """
    Returns if the provided mask is a dummy mask, consisting of only -1 values.

    :param mask: Object mask
    :return: True, if the object mask is a dummy box, False otherwise
    """
    return torch.all(mask.eq(-1))


def pad(t, to_size):
    """
    Pads the given tensor to the provided size. Padding is always applied to the second last dimension.

    :param t: A tensor
    :param to_size: Target size, that has to be reached
    :return: A padded version of t
    """
    current_size = t.size()[0]
    if current_size > to_size:
        raise RuntimeError("Cannot pad to a smaller size!")
    if current_size == to_size:
        return t
    # Get dimensions
    dim = len(t.size())
    current_size = t.size()[0]
    pad_vector = (0,)
    for i in range(1, dim):
        pad_vector += (0, 0,)
    pad_vector += (to_size - current_size,)
    t = F.pad(t, pad_vector, mode='constant', value=-1)
    return t


def tensor_to_cv2_image(t):
    """
    Transform the given PyTorch tensor into a cv2 image.

    :param t: Tensor of size (3,H,W)
    :return: cv2 image
    """
    t = t.cpu()
    return np.array(T.ToPILImage()(t))


def get_uuid():
    """
    Returns a new UUID.

    :return: UUID
    """
    return str(uuid.uuid4())


def read_ranges(string):
    """
    Parses parameter ranges (for hyperparameter tuning) from the given YAML string.
    
    Structure: <parameter_name>_range : (<lower_bound>,<upper_bound>)
    
    :param string: YAML string
    :return: Dict with the parameter ranges
    """
    if string is None or not string:
        return {}
    parameter_dict = yaml.safe_load(string)
    ret = {}
    for parameter_name in parameter_dict.keys():
        if parameter_name.endswith("_range"):
            # Extract parameter name from the key
            # Lower and upper bound are represented as tuple
            ret[parameter_name[:-6]] = tuple(parameter_dict[parameter_name])
        else:
            ret[parameter_name] = parameter_dict[parameter_name]
    return ret


def get_root_dataset(dataset):
    """
    Returns the root (main) dataset for the given dataset, which can be a subset of the root dataset.
    
    :param dataset: Some dataset
    :return: The root dataset for the given dataset
    """

    d = dataset
    tries = 5
    while isinstance(d, Subset) and tries > 0:
        d = d.dataset
        tries -= 1

    if isinstance(d, Subset):
        raise RuntimeError("Could not find root dataset!")

    return d


def get_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes

    :param box1: Bounding box coordinates in the form (x0,y0,x1,y1)
    :param box2: Bounding box coordinates in the form (x0,y0,x1,y1)
    :return: The IoU value of the two provided bounding boxes
    """
    box1 = box1.float().cpu()
    box2 = box2.float().cpu()

    x0 = torch.max(box1[0], box2[0])
    y0 = torch.max(box1[1], box2[1])
    x1 = torch.min(box1[2], box2[2])
    y1 = torch.min(box1[3], box2[3])

    intersection_width = torch.clamp(x1 - x0 + 1, min=0)
    intersection_height = torch.clamp(y1 - y0 + 1, min=0)

    intersection_area = intersection_width * intersection_height
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = union_area - intersection_area
    iou = intersection_area / union_area
    return iou


def get_spatial_mask(box, image_h, image_w, mask_size=None):
    """
    Creates a binary spatial mask for the given box.

    :param box: Bounding box coordinates of the form (x0,y0,x1,y1)
    :param image_h: Height of the image the box occurs in
    :param image_w: Width of the image the box occurs in
    :param mask_size: Optional. Side length of the target spatial mask. If not provided,
    the spatial mask is not rescaled
    :return: The spatial mask
    """

    mask = np.zeros((image_h, image_w, 1))
    cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), 1, cv2.FILLED)
    mask_t = torch.tensor(mask)

    # Scale down mask
    if mask_size is not None:
        mask_t = mask_t.view((1, 1, image_h, image_w))
        mask_t = F.interpolate(mask_t, (mask_size, mask_size))
        mask_t = mask_t[0, 0].flatten()

    mask_t = mask_t.float()
    mask_t = mask_t.to(get_device())
    return mask_t


def expand_boxes(all_boxes, box_expansion_factor, image_height, image_width):
    """
    Expands the given bounding boxes by the provided factor.

    N: Batch size
    U: Upper bound for the number of objects (padding size)

    :param all_boxes: Tensor of bounding boxes coordinates of size (N,U,4)
    :param box_expansion_factor: Expansion factor
    :param image_height: Height of the image the boxes occur in
    :param image_width: Width of the image the boxes occur in
    :return: A tensor of size (N,U,4) with the expanded bounding boxes
    """
    ret = torch.empty(all_boxes.size(), device=get_device())
    for i, boxes in enumerate(all_boxes):
        for j, box in enumerate(boxes):
            if is_dummy_box(box):
                ret[i][j] = box
                continue
            box = box.clone()
            width = box[2] - box[0]
            height = box[3] - box[1]

            # Expand box
            box[0] = max(box[0] - 0.5 * box_expansion_factor * width, 0)
            box[2] = min(box[2] + 0.5 * box_expansion_factor * width, image_width)

            box[1] = max(box[1] - 0.5 * box_expansion_factor * height, 0)
            box[3] = min(box[3] + 0.5 * box_expansion_factor * height, image_height)

            ret[i][j] = box

    return ret
