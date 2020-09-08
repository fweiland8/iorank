import os
import random

import logging
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Subset

import iorank.util.util as util


def read_labels_boxes(filename, label_mapping):
    """

    Reads class labels and bounding box coordinates from the file with the provided filename.

    :param filename: Filename of the file containing labels and bounding box coordinates
    :param label_mapping: Mapping from numerical label id to label name
    :return: tuple (labels,boxes)
    """
    labels = []
    boxes = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        s = line.split(' ')
        # DontCare objects, i.e. small, not-identifiable background objects are not considered
        if s[0] == 'DontCare':
            continue
        box = s[4:8]
        label = s[0]
        labels.append(label_mapping[label])
        boxes.append(np.array(box, dtype=float))
    return labels, boxes


def mean_depth_for_box(depthmap, box):
    """
    Computes the mean depth value for a given bounding box and depth map

    :param depthmap: Depth map with depth values
    :param box: Bounding box coordinates (x0,y0,x1,y1)
    :return: Average depth value inside the provided box
    """
    box = box.astype(int)
    roi = depthmap[box[1]:box[3] + 1, box[0]:box[2] + 1]
    roi_values = roi.flatten()
    mean = roi_values[roi_values > 0].mean()
    return mean


def read_depthmap(filename):
    """
    Reads a depth map from the file with the provided filename.

    Code is taken from the official KITTI Depth Development Kit:
    http://www.cvlibs.net/downloads/depth_devkit.zip

    :param filename: Filename of the depth map file
    :return: The depth map
    """
    depth_png = np.array(Image.open(filename), dtype=int)
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth


def get_boxes_path(root, image_id):
    """
    Return the path to the bounding box file for a given image id

    :param root: Dataset root directory
    :param image_id: Image Id
    :return: Path to the bounding box file for the given image id
    """
    return os.path.join(root, "boxes", image_id + ".txt")


def get_image_id(image_filename):
    """
    Extracts the image id from the provided filename.

    :param image_filename: The filename
    :return: The image id
    """
    return os.path.splitext(image_filename)[0]


def shrink_to_ranking_size(boxes, depths, labels, ranking_size):
    """
    Shrinks the given bounding boxes, depth values and class labels to the provided ranking size.

    The top 'ranking size' objects according to their bounding box size are retained.

    :param boxes: List of bounding box coordinates
    :param depths: List of depth values
    :param labels: List of class labels
    :param ranking_size: Target size of the lists
    :return: (boxes,depths,labels) lists shrinked to the provided ranking size
    """
    sizes = np.array([util.get_box_size(box) for box in boxes])
    sizes_to_take = np.argsort(sizes)[-ranking_size:]
    boxes = boxes[sizes_to_take]
    depths = depths[sizes_to_take]
    labels = labels[sizes_to_take]
    return boxes, depths, labels


class KittiDataset(Dataset):
    def __init__(self, root, mode=None, min_ranking_size=2, max_ranking_size=None, in_memory=False, augmentation=False):
        """
        Create an instance of the KITTI dataset.

        :param root: Root folder of the dataset on the file system
        :param mode: Either 'train' or 'test'
        :param min_ranking_size: Minimum number of objects that have to present. Images with fewer objects are filtered out.
        :param max_ranking_size: Maximum number of objects that have to present. Images with more objects are filtered out.
        :param in_memory: Determines if the dataset has to be kept in memory
        :param augmentation: Enables or disables data augmentation
        """

        self.logger = logging.getLogger(KittiDataset.__name__)
        self.logger.info("Creating dataset")

        if mode is None:
            raise RuntimeError("Mode out of (train,test) must be provided!")

        if mode != "train" and mode != "test":
            raise RuntimeError("Invalid mode: {}".format(mode))

        self.root = os.path.join(root, "KITTI", mode)
        self.in_memory = in_memory
        self.min_ranking_size = min_ranking_size
        self.max_ranking_size = max_ranking_size
        self.augmentation = augmentation

        self.device = util.get_device()

        self.label_mapping = {
            "Background": 0,
            "Car": 1,
            "Van": 2,
            "Truck": 3,
            "Pedestrian": 4,
            "Cyclist": 5,
            "Misc": 6,
            "Person_sitting": 7,
            "Tram": 8
        }

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

            # Consider only images with more than 'min_ranking_size' objects
            if len(target["labels"]) >= min_ranking_size:
                self.image_filenames.append(image_filename)
                if drive_id not in self.drive_to_idx_mapping.keys():
                    self.drive_to_idx_mapping[drive_id] = []
                self.drive_to_idx_mapping[drive_id].append(idx)
                idx += 1

                # Store image and targets in memory
                if self.in_memory:
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

        :return: Number of classes in the dataset.
        """
        return len(self.label_mapping)

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
        Returns image at the given index, either from memory or from file system.

        :param idx: Index
        :return: Image
        """
        if self.in_memory:
            image = self.images[idx]
        else:
            image = self._get_pil_image(self.image_filenames[idx])
        return image

    def _get_target(self, idx):
        """
        Reads the target for the given index, either from memory or from file system.
        :param idx: Index
        :return: Dict with 'boxes', 'labels' and 'scores'
        """
        if self.in_memory:
            return self.targets[idx]

        image_id = get_image_id(self.image_filenames[idx])
        target = self._read_target(image_id)
        return target

    def _read_target(self, image_id):
        """
        Read target for the given image id from file system.

        :param image_id: The image id
        :return: Dict with 'boxes', 'labels' and 'scores'
        """
        boxes_filename = os.path.join(self.root, "boxes", image_id + ".txt")
        labels, boxes = read_labels_boxes(boxes_filename, self.label_mapping)
        depthmap_filename = os.path.join(self.root, "depthmaps", image_id + ".png")
        depthmap = read_depthmap(depthmap_filename)
        depths = [mean_depth_for_box(depthmap, box) for box in boxes]

        if True in np.isnan(depths):
            invalid_indices = np.where(np.isnan(depths))[0]
            labels = [labels[i] for i in range(len(labels)) if i not in invalid_indices]
            boxes = [boxes[i] for i in range(len(boxes)) if i not in invalid_indices]
            depths = [depths[i] for i in range(len(depths)) if i not in invalid_indices]

        return {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels), "scores": torch.tensor(depths)}

    def _prepare_target(self, target):
        """
        Prepares the given target data. Especially, padding is applied here in order to have tensors of unified size.

        :param target: Raw target data
        :return: Padded target data
        """
        labels = target["labels"]
        depths = target["scores"]
        boxes = target["boxes"]

        if self.max_ranking_size is not None and len(boxes) > self.max_ranking_size:
            boxes, depths, labels = shrink_to_ranking_size(boxes, depths, labels, self.max_ranking_size)

        conf = torch.ones(boxes.size(0))

        # Padding
        boxes = util.pad(boxes, self.padding_size)
        depths = util.pad(depths, self.padding_size)
        labels = util.pad(labels, self.padding_size)
        conf = util.pad(conf, self.padding_size)

        target = {"boxes": boxes, "scores": depths, "labels": labels, "conf": conf}
        return target

    def _get_item(self, idx):
        """
        Returns dataset item at the given index.

        :param idx: Index
        :return: Dataset item
        """
        if self.augmentation:
            image, target = self._get_image(idx), self._get_target(idx)
            image, target = self._augment(image, target)
            target = self._prepare_target(target)
            return image, target
        else:
            image, target = self._get_image(idx), self._get_target(idx)
            image = TF.resize(image, (self.h, self.w))
            image = TF.to_tensor(image)
            target = self._prepare_target(target)
            return image, target

    def _augment(self, image, t):
        """
        Applies data augmentation to the provided image and target data.

        :param image: The image to be augmented
        :param t: Corresponding ground truth data
        :return: An augmented image with adjusted ground truth data
        """
        image_orig = image
        boxes_orig = t["boxes"]

        image = image.copy()

        boxes = t["boxes"].clone()
        labels = t["labels"].clone()
        scores = t["scores"].clone()

        valid_boxes = []

        # Scale image to common size in any case 
        image = TF.resize(image, (self.h, self.w))

        # Translation
        movement_horizontal = random.randint(-100, 100)
        movement_vertical = random.randint(-50, 50)
        image = TF.affine(image, angle=0, translate=(movement_horizontal, movement_vertical), scale=1.0, shear=0.0)

        # Adjust bounding boxes after translation
        for i, box in enumerate(boxes):
            box[0] = torch.clamp(box[0] + movement_horizontal, min=0, max=self.w)
            box[2] = torch.clamp(box[2] + movement_horizontal, min=0, max=self.w)

            box[1] = torch.clamp(box[1] + movement_vertical, min=0, max=self.h)
            box[3] = torch.clamp(box[3] + movement_vertical, min=0, max=self.h)

            # Box is (almost) entirely out of the image
            if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                self.logger.warning("Ignoring invalid box %s", box)
            else:
                valid_boxes.append(i)

        if len(valid_boxes) == 0:
            self.logger.warning("No box is left after translation. Reverting translation..")
            boxes = boxes_orig.clone()
            image = image_orig.copy()
        else:
            boxes = boxes[valid_boxes]
            labels = labels[valid_boxes]
            scores = scores[valid_boxes]

        # Adjust color values
        t = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0)
        image = t(image)

        # Flip image with probability 0.5
        if random.random() > 0.5:
            image = TF.hflip(image)
            for box in boxes:
                box[0] = box[0] + 2 * (0.5 * self.w - box[0])
                box[2] = box[2] + 2 * (0.5 * self.w - box[2])

                w = abs(box[2] - box[0])

                box[0] = box[0] - w
                box[2] = box[2] + w

        # Finally, turn PIL image to PyTorch tensor
        image = TF.to_tensor(image)
        target = {"boxes": boxes, "labels": labels, "scores": scores}
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
    # We need root dataset in order to get the drive-to-idx mapping
    if isinstance(dataset, Subset):
        indices = dataset.indices

        limit = 5
        while not isinstance(dataset, KittiDataset) and limit > 0:
            dataset = dataset.dataset
            limit -= 1

    drive_idx_mapping = dataset.get_drive_to_idx_mapping(indices)

    drive_ids = list(drive_idx_mapping.keys())

    if shuffle:
        random.shuffle(drive_ids)

    indices1 = []
    indices2 = []

    # Split on drive level, i.e. data items of a particular drive are either completely in the first or the second
    # subset.
    for drive_id in drive_ids:
        indices = drive_idx_mapping[drive_id]
        l = len(indices)
        if len(indices2) + l <= split2_size:
            indices2.extend(indices)
        else:
            indices1.extend(indices)

    return indices1, indices2
