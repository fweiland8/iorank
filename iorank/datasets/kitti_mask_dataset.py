import os
import random

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Subset

import iorank.util.util as util


def resize(t, h, w):
    """
    Resizes the given image.

    :param t: Image, given as tensor
    :param h: Target height
    :param w: Target width
    :return: The resized image
    """

    t = t.view(1, 1, t.size(0), t.size(1))
    t = F.interpolate(t, size=(h, w))
    t = t[0, 0]
    return t


def mean_depth_for_mask(depthmap, mask):
    """
    Computes the average depth inside the given object mask.

    :param depthmap: Depthmap with depth values
    :param mask: Object mask
    :return: The mean depth
    """

    if depthmap.size() != mask.size():
        raise RuntimeError("Depthmap and mask need to be of the same size")

    # Reduce depthmap to the mask area
    mask_depths = depthmap * mask

    n_depth_values = len(mask_depths[mask_depths > 0])

    # Return invalid value
    if n_depth_values == 0:
        return torch.tensor(-1.0).double()

    mean_depth = torch.sum(mask_depths[mask_depths > 0]) / n_depth_values
    return mean_depth


def box_for_mask(mask):
    """
    Creates bounding box coordinates for the given object mask.

    :param mask: Object mask
    :return: Bounding box coordinates of the form (x0,y0,x1,y1)
    """

    y, x = torch.where(mask)
    x_min = torch.min(x)
    x_max = torch.max(x)
    y_min = torch.min(y)
    y_max = torch.max(y)
    box = torch.stack([x_min, y_min, x_max, y_max])
    return box


def get_image_id(image_filename):
    """
    Extracts the image id from the provided filename.

    :param image_filename: The filename
    :return: The image id
    """
    return os.path.splitext(image_filename)[0]


class KittiMaskDataset(Dataset):
    def __init__(self, root, mode=None, min_ranking_size=2, augmentation=False, **kwargs):
        """
        Create an instance of the KITTIMask dataset, which is a special variant of the KITTI dataset, where object
        masks instead of bounding boxes are used.

        :param root: Root folder of the dataset on the file system
        :param mode: Either 'train' or 'test'
        :param min_ranking_size: Minimum number of objects that have to present. Images with fewer objects are filtered out.
        :param augmentation: Enables or disables data augmentation
        :param kwargs:
        """

        self.logger = logging.getLogger(KittiMaskDataset.__name__)
        self.logger.info("Creating dataset")

        if mode is None:
            raise RuntimeError("Mode out of (train,test) must be provided!")

        if mode != "train" and mode != "test":
            raise RuntimeError("Invalid mode: {}".format(mode))

        self.root = os.path.join(root, "KITTI", mode)
        # Mask dataset is always inmemory
        self.in_memory = True
        self.min_ranking_size = min_ranking_size
        self.augmentation = augmentation

        self.device = util.get_device()
        self.padding_size = 22

        self.h = 375
        self.w = 1242

        self.image_filenames = []
        self.images = []
        self.targets = []
        self.drive_to_idx_mapping = {}
        self.logger.info("Reading directory %s, preloading = %s", self.root, self.in_memory)
        idx = 0
        for image_filename in list(sorted(os.listdir(os.path.join(self.root, "images")))):
            image_id = get_image_id(image_filename)
            drive_id = image_id.split("_")[1]

            target = self._read_target(image_id)

            if target is not None and len(target["masks"]) >= min_ranking_size:
                self.image_filenames.append(image_filename)
                if drive_id not in self.drive_to_idx_mapping.keys():
                    self.drive_to_idx_mapping[drive_id] = []
                self.drive_to_idx_mapping[drive_id].append(idx)
                idx += 1

                self.images.append(self._get_pil_image(image_filename))
                self.targets.append(target)
        self.logger.info("Finished creating dataset")

    def __getitem__(self, idx):
        """
        Returns dataset items. If an integer is provided, the dataset item at the given index is returend. If a list
        of indices is provided, a corresponding list of dataset items is returned.

        :param idx: Single index or list of indices
        :return: The dataset items
        """
        if isinstance(idx, list):
            items = []
            for index in idx:
                items.append(self._get_item(index))
            return items
        else:
            return self._get_item(idx)

    def __len__(self):
        """
        Return the length of the dataset.

        :return: Length of the dataset.
        """
        return len(self.image_filenames)

    def get_n_classes(self):
        """
        Returns the number of classes in the dataset.

        In the KITTI mask dataset there are no classes, so 1 is returned.

        :return: Number of classes in the dataset.
        """
        return 1

    def _get_pil_image(self, image_filename):
        """
        Reads an image from the given file

        :param image_filename: File to read image from
        :return: PIL image
        """
        image_path = os.path.join(self.root, "images", image_filename)
        image = Image.open(image_path).convert("RGB")
        return image

    def _get_image(self, idx):
        """
        Returns image at the given index from memory.

        :param idx: Index
        :return: Image
        """
        return self.images[idx]

    def _get_target(self, idx):
        """
        Reads the target for the given index from memory.
        :param idx: Index
        :return: Dict with 'boxes', 'masks', and 'scores'
        """
        return self.targets[idx]

    def _read_target(self, image_id):
        """
        Read target for the given image id from file system.

        :param image_id: The image id
        :return: Dict with 'boxes', 'masks' and 'scores'
        """
        masks_filename = os.path.join(self.root, "masks", image_id + ".png")
        masks = self.read_masks(masks_filename)
        depthmap_filename = os.path.join(self.root, "depthmaps", image_id + ".png")
        depthmap = self._read_depthmap(depthmap_filename)
        depths = [mean_depth_for_mask(depthmap, mask) for mask in masks]
        boxes = [box_for_mask(mask) for mask in masks]

        if True in np.isnan(depths):
            invalid_indices = np.where(np.isnan(depths))[0]
            boxes = [boxes[i] for i in range(len(boxes)) if i not in invalid_indices]
            depths = [depths[i] for i in range(len(depths)) if i not in invalid_indices]
            masks = [masks[i] for i in range(len(masks)) if i not in invalid_indices]

        if len(boxes) > 0:
            target = {"boxes": torch.stack(boxes), "masks": torch.stack(masks), "scores": torch.stack(depths)}

            # Store masks in sparse tensor for efficient in memory storage
            target["masks"] = target["masks"].to_sparse()
            return target
        else:
            None

    def _prepare_target(self, target):
        """
        Prepares the given target data. Especially, padding is applied here in order to have tensors of unified size.

        :param target: Raw target data
        :return: Padded target data
        """
        masks = target["masks"]
        depths = target["scores"]
        boxes = target["boxes"]

        # Turn masks to dense version
        masks = masks.to_dense()

        conf = torch.ones(boxes.size(0))
        labels = torch.ones(boxes.size(0))

        # Padding
        boxes = util.pad(boxes, self.padding_size)
        depths = util.pad(depths, self.padding_size)
        masks = util.pad(masks, self.padding_size)
        conf = util.pad(conf, self.padding_size)
        labels = util.pad(labels, self.padding_size)

        target = {"boxes": boxes, "scores": depths, "masks": masks, "conf": conf, "labels": labels}
        return target

    def _get_item(self, idx):
        """
       Returns dataset item at the given index.

       :param idx: Index
       :return: Dataset item
       """
        if self.augmentation:
            # Augmentation is not yet implemented for the mask dataset
            raise NotImplementedError("Augmentation is not yet supported")
        else:
            image, target = self._get_image(idx), self._get_target(idx)
            image = TF.resize(image, (self.h, self.w))
            image = TF.to_tensor(image)
            target = self._prepare_target(target)
            return image, target

    def get_drive_to_idx_mapping(self, indices=None):
        """
        Returns a mapping from (car) drive ids to image indices

        :param indices: Optional parameter, reduces mapping to the provided indices
        :return: Mapping from (car) drive ids to image indices
        """
        if indices is None:
            return self.drive_to_idx_mapping

        mapping = self.drive_to_idx_mapping
        mapping = {drive_id: [idx for idx in mapping[drive_id] if idx in indices] for drive_id in
                   self.drive_to_idx_mapping.keys()}
        mapping = {k: v for k, v in mapping.items() if len(v) > 0}
        return mapping

    def _read_depthmap(self, filename):
        """
        Reads a depth map from the file with the provided filename.

        Code is taken from the official KITTI Depth Development Kit:
        http://www.cvlibs.net/downloads/depth_devkit.zip

        :param filename: Filename of the depth map file
        :return: The depth map
        """
        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)

        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.

        t = torch.tensor(depth)
        t = resize(t, self.h, self.w)

        return t

    def read_masks(self, mask_file):
        """
        Read the object masks from the given file.

        :param mask_file: Path to the file containing the object masks
        :return: List of object masks
        """
        masks_image = Image.open(mask_file)
        arr = np.array(masks_image)
        t = torch.tensor(arr, dtype=torch.float)
        t = resize(t, self.h, self.w)

        object_ids = torch.unique(t)

        all_masks = [torch.eq(t, object_id).int() for object_id in object_ids if object_id != 0]
        return all_masks


def split(dataset, fraction, shuffle=True):
    """
    Splits the given dataset into two subsets of size 'fraction' and '1-fraction'.

    :param dataset: Dataset to split
    :param fraction: Fraction of one of the target subsets
    :param shuffle: If true, items are shuffled before splitting. Default: True
    :return:
    """
    full_size = len(dataset)
    split2_size = int(fraction * full_size)

    indices = None
    if isinstance(dataset, Subset):
        indices = dataset.indices

        limit = 5
        while not isinstance(dataset, KittiMaskDataset) and limit > 0:
            dataset = dataset.dataset
            limit -= 1

    drive_idx_mapping = dataset.get_drive_to_idx_mapping(indices)

    drive_ids = list(drive_idx_mapping.keys())

    if shuffle:
        random.shuffle(drive_ids)

    indices1 = []
    indices2 = []

    for drive_id in drive_ids:
        indices = drive_idx_mapping[drive_id]
        l = len(indices)
        if len(indices2) + l <= split2_size:
            indices2.extend(indices)
        else:
            indices1.extend(indices)

    return indices1, indices2
