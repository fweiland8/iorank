import os

import argparse
import pathlib
import shutil

"""
Dataset converter for the KITTI dataset.

"""

train_instances_filename = "kitti_train.txt"
test_instances_filename = "kitti_test.txt"


def create_dataset(mode, instances_file, boxes_dir, images_dir, depthmaps_dir, kitti_dir):
    print("Create {} set".format(mode))

    mode_dir = os.path.join(kitti_dir, mode)
    if not os.path.exists(mode_dir):
        os.mkdir(mode_dir)

    images_out_dir = os.path.join(mode_dir, "images")
    if not os.path.exists(images_out_dir):
        os.mkdir(images_out_dir)

    boxes_out_dir = os.path.join(mode_dir, "boxes")
    if not os.path.exists(boxes_out_dir):
        os.mkdir(boxes_out_dir)

    depthmaps_out_dir = os.path.join(mode_dir, "depthmaps")
    if not os.path.exists(depthmaps_out_dir):
        os.mkdir(depthmaps_out_dir)

    with open(instances_file) as f:
        entries = f.readlines()

    n_instances = len(entries)

    for i, entry in enumerate(entries):
        entry = entry.strip()
        split = entry.split(";")
        image_id = split[0]
        boxes_id = split[1]
        drive_date = split[2]
        drive_id = split[3]
        image_no = split[4]

        # Required depth map can be in train or val folder
        depthmap_file_train = os.path.join(depthmaps_dir, "train", drive_id, "proj_depth", "groundtruth", "image_02",
                                           image_no + ".png")
        depthmap_file_val = os.path.join(depthmaps_dir, "val", drive_id, "proj_depth", "groundtruth", "image_02",
                                         image_no + ".png")

        if os.path.exists(depthmap_file_train):
            depthmap_file = depthmap_file_train
        elif os.path.exists(depthmap_file_val):
            depthmap_file = depthmap_file_val
        else:
            raise RuntimeError("Cannot find depthmap for image id {}".format(image_id))

        boxes_file = os.path.join(boxes_dir, "training", "label_2", boxes_id + ".txt")
        if not os.path.exists(boxes_file):
            raise RuntimeError("Missing boxes file: {}".format(boxes_file))

        image_file = os.path.join(images_dir, drive_date, drive_id, "image_02", "data", image_no + ".png")
        if not os.path.exists(image_file):
            raise RuntimeError("Missing image file: {}".format(image_file))

        image_target_file = os.path.join(images_out_dir, image_id + ".png")
        depthmap_target_file = os.path.join(depthmaps_out_dir, image_id + ".png")
        boxes_target_file = os.path.join(boxes_out_dir, image_id + ".txt")

        shutil.copy2(depthmap_file, depthmap_target_file)
        shutil.copy2(image_file, image_target_file)
        shutil.copy2(boxes_file, boxes_target_file)

        print("({}/{})  Copied data for image with id {} to {} directory".format(i + 1, n_instances, image_id, mode))


# Parse arguments
parser = argparse.ArgumentParser(description='KITTI dataset converter. Reorganizes the downloadable KITTI data such '
                                             'that they can be used from the iorank framework.')
parser.add_argument('-i', '--images_dir', help='Path to the raw images directory', required=True)
parser.add_argument('-b', '--boxes_dir', help='Path to the directory containing boxes/labels', required=True)
parser.add_argument('-d', '--depthmaps_dir', help='Path to the depthmap directory', required=True)
parser.add_argument('-o', '--output_dir', help='Path under which the KITTI dataset will be available', required=True)
args = vars(parser.parse_args())

images_dir = args["images_dir"]
boxes_dir = args["boxes_dir"]
depthmaps_dir = args["depthmaps_dir"]
output_dir = args["output_dir"]

current_dir = pathlib.Path(__file__).parent.absolute()

train_instances_file = os.path.join(current_dir, train_instances_filename)
test_instances_file = os.path.join(current_dir, test_instances_filename)

if not os.path.isfile(train_instances_file):
    raise RuntimeError("Missing file: {}".format(train_instances_file))

if not os.path.isfile(test_instances_file):
    raise RuntimeError("Missing file: {}".format(test_instances_file))

if not os.path.exists(output_dir):
    print("Creating output directory")
    os.mkdir(output_dir)

kitti_dir = os.path.join(output_dir, "KITTI")
print("Dataset will be available at {}".format(kitti_dir))
if not os.path.exists(kitti_dir):
    os.mkdir(kitti_dir)

create_dataset("train", train_instances_file, boxes_dir, images_dir, depthmaps_dir, kitti_dir)
create_dataset("test", test_instances_file, boxes_dir, images_dir, depthmaps_dir, kitti_dir)
print("Finished successfully")
