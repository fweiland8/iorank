import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import logging


def visualize_image_with_depth(img, d_map):
    plt.imshow(img)
    plt.show()

    plt.imshow(d_map, cmap='gray')
    plt.show()


# Assumption: argsorted preds
def visualize_depth_map_prediction(segments, preds):
    assert len(segments) == len(preds)

    d_map = np.zeros([segments[0].get_mask.shape[0], segments[0].get_mask.shape[1]])

    step = 255 / len(preds)

    for i in range(len(preds)):
        # Since the prediction is sometimes in between natural numbers (e.g. RankSVM can produce such rankings), the
        # nearest number is chosen
        pred_idx = int(round(preds[i]))

        mask = segments[i].get_mask
        if len(mask.shape) == 3:
            mask = np.any(mask, axis=-1)
        d_map[mask] = pred_idx * step

    plt.imshow(d_map, cmap='gray_r')
    plt.show()


def visualize_depth_map_prediction2(preds, shape):
    preds_reshaped = preds.reshape(shape)
    preds_reshaped = ((preds_reshaped - np.min(preds_reshaped)) * 255) / (
                np.max(preds_reshaped) - np.min(preds_reshaped))

    plt.imshow(preds_reshaped, cmap='gray')
    plt.show()


def visualize_depth_features(segments, segment_features, title=""):
    assert len(segments) == segment_features.shape[
        0], "The number of segments (actual: {}) must match the number of segments for which features are available " \
            "(actual: {})!".format(len(segments), segment_features.shape[0])

    result_map = np.zeros([segments[0].get_mask.shape[0], segments[0].get_mask.shape[1]])

    for i in range(len(segments)):
        mask = segments[i].get_mask
        if len(mask.shape) == 3:
            mask = np.any(mask, axis=-1)
        result_map[mask] = segment_features[i]

    # Scale result map
    max_res_map = np.max(result_map)
    result_map /= max_res_map

    plt.title(title)
    plt.imshow(result_map, cmap='gray_r', vmax=1, vmin=0)
    plt.show()


def visualize_single_img(img, title=""):
    plt.title(title)
    plt.imshow(img)
    plt.show()


def visualize_single_map(s_map, title=""):
    s_map /= (np.max(s_map) + 0.0001)
    plt.title(title)
    plt.imshow(s_map, cmap='gray_r', vmax=1, vmin=0)
    plt.show()


def draw_lines_in_image(image, lines, show_image=False, write_img_path=None):
    image = np.copy(image)

    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i].x1, lines[i].y1, lines[i].x2, lines[i].y2
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        cv.line(image, (y1, x1), (y2, x2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)

    if show_image:
        plt.imshow(image)
        plt.show()

    if write_img_path is not None:
        cv.imwrite(write_img_path, image)

    return image


def plot_points_in_image(image, points, show_image=False, write_img_path=None, color=np.array([255, 0, 0])):
    image = np.copy(image)

    for p in points:
        p0 = int(round(p[0]))
        p1 = int(round(p[1]))

        if p0 >= image.shape[0] or p0 < 0 or p1 >= image.shape[1] or p1 < 0:
            break

        image[p0, p1] = color

    logging.debug("Showing image...")

    if show_image:
        plt.imshow(image)
        plt.show()

    if write_img_path is not None:
        cv.imwrite(write_img_path, image)

    return image


def plot_v_points(image, v_points, color=np.array([255, 0, 0])):
    image = np.copy(image)

    for v_p in v_points:

        p0_vp = [max(0, round(v_p._v_point[0] -5)),
                 min(image.shape[0], round(v_p._v_point[0] + 5))]
        p1_vp = [max(0, round(v_p._v_point[1] -5)),
                 min(image.shape[1], round(v_p._v_point[1] + 5))]

        image[int(p0_vp[0]):int(p0_vp[1]), int(p1_vp[0]):int(p1_vp[1])] = color

        p0_bl = v_p._border_line.support_vector + 100 * v_p._border_line.direction
        p1_bl = v_p._border_line.support_vector - 0 * v_p._border_line.direction
        x1 = int(round(p0_bl[0]))
        y1 = int(round(p0_bl[1]))
        x2 = int(round(p1_bl[0]))
        y2 = int(round(p1_bl[1]))
        cv.line(image, (y1, x1), (y2, x2), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)


    logging.debug("Showing image...")
    plt.imshow(image)
    plt.show()
