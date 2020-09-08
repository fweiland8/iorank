from torch.utils.data import Subset

from iorank.datasets import kitti_dataset, kitti_mask_dataset
from iorank.datasets.kitti_dataset import KittiDataset
from iorank.datasets.kitti_mask_dataset import KittiMaskDataset


def get_split(dataset):
    """
    Returns the split function for the given dataset.

    :param dataset: The dataset for which the split function has to be determined
    :return: Suitable split function for the provided dataset
    """

    # First, the root dataset has to be found
    limit = 5
    while isinstance(dataset, Subset) and limit > 0:
        dataset = dataset.dataset
        limit -= 1
    if limit == 0:
        raise RuntimeError("Could not find root dataset")

    # Return split function dependent on the dataset
    if isinstance(dataset, KittiDataset):
        return kitti_dataset.split
    elif isinstance(dataset, KittiMaskDataset):
        return kitti_mask_dataset.split
    else:
        raise RuntimeError("Could not find split for dataset {}".format(dataset.__class__.__name__))
