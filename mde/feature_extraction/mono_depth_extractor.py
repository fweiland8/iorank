import cv2 as cv
import logging
import numpy as np
from tqdm import tqdm

from mde.feature_extraction import color_utils
from mde.feature_extraction.color_utils import get_saturation
from mde.feature_extraction.feature_extractor import FeatureExtractor
from mde.feature_extraction.shape_utils import realign_line, detect_hough_lines, \
    line_fusion_filter, line_length_filter, line_angle_filter
from mde.feature_extraction.shapes import Line
from mde.globalization.scaler import mask_array_varying_size
from mde.math_utils import euclidean_distance
from mde.viz.viz_utils import draw_lines_in_image, visualize_single_map, plot_v_points


class RelativeHeightFeatureExtractor(FeatureExtractor):
    def extract_features(self, segments, image):
        if len(segments) < 1:
            logging.warning("Warning: Empty input is given to feature extractor.")
            return None

        result_features = np.zeros(len(segments))

        im_height = image.shape[0]

        for seg_idx in range(len(segments)):
            avg_y_coord, _ = segments[seg_idx].get_center

            result_features[seg_idx] = 1 - ((im_height - 1) - avg_y_coord) / (im_height - 1)

        return result_features


class ContourFeatureExtractor(FeatureExtractor):
    def extract_features(self, segments, image):
        if len(segments) < 1:
            logging.warning("Warning: Empty input is given to feature extractor.")
            return None

        result_features = np.zeros(len(segments))

        for seg_idx in range(len(segments)):
            # Assume RGB images with values in [0, 255]
            # seg_im = images[i][segments[i][inst]]
            # seg_im = crop_segment(segments[seg_idx].get_mask, image)
            seg_im = segments[seg_idx].get_img_crop(image)

            imgray = cv.cvtColor(seg_im.astype('uint8'), cv.COLOR_RGB2GRAY)
            # TODO: Determine correct threshold (200 has been used in paper implementation)
            ret, thres = cv.threshold(imgray, 200, 255, 0)

            # Resulting contours is list of numpy arrays storing (x,y) pairs determining the contours boundaries
            contours, hierarchy = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            result_features[seg_idx] = len(contours) / (seg_im.shape[0] * seg_im.shape[1])
        return result_features


class AtmosphericPerspectiveFeatureExtractor(FeatureExtractor):
    def extract_features(self, segments, image):
        if len(segments) < 1:
            logging.warning("Warning: Empty input is given to feature extractor.")
            return None

        result_features = np.zeros(len(segments))

        for seg_idx in range(len(segments)):
            seg_im = segments[seg_idx].get_img_crop(image)

            sats = get_saturation(seg_im)
            result_features[seg_idx] = 1 - np.mean(sats)

        return result_features


class LinearPerspectiveFeatureExtractor(FeatureExtractor):
    def __init__(self, target_height=None, target_width=None, verbose=1):
        self.verbose = verbose
        self.target_height = target_height
        self.target_width = target_width

    class VanishingPoint:
        def __init__(self, line1, line2, im_height, im_width):
            self.line1 = line1
            self.line2 = line2
            self.im_height = im_height
            self.im_width = im_width
            self._build()

        def _build(self):
            self._confidence_range = max(self.im_height, self.im_width) / 2.
            self._v_point = self.line1.intersection(self.line2)

            self.line1 = realign_line(self.line1, self._v_point)
            self.line2 = realign_line(self.line2, self._v_point)

            if not np.isinf(self._v_point).any():
                bisection = self.line1.get_bisection(self.line2)
                self._border_line = bisection.get_orthogonal()
                self._border_line.invert_direction()

                self._max_dist_2_border = self._get_max_dist_to_border()

                # TODO: Detect overlapping and fix
                if self.line1.get_distance_to_vec(self.line2.support_vector) < 0:
                    self.line1.invert_direction()
                if self.line2.get_distance_to_vec(self.line1.support_vector) < 0:
                    self.line2.invert_direction()

        @property
        def get_v_point(self):
            return self._v_point

        def _point_in_image(self, point):
            return (point >= 0).all() and point[0] <= self.im_height and point[1] <= self.im_width

        def _get_image_border_intersection(self, line):
            # Calculate image border lines
            left = Line(np.array([0, 0]), np.array([1, 0]))
            bottom = Line(np.array([self.im_height, 0]), np.array([0, 1]))
            right = Line(np.array([self.im_height, self.im_width]), np.array([-1, 0]))
            up = Line(np.array([0, self.im_width]), np.array([0, -1]))

            # Calculate (infinite) intersection with line
            new_line = Line(self._v_point, line.direction)

            int_up = new_line.intersection(up)
            int_left = new_line.intersection(left)
            int_bottom = new_line.intersection(bottom)
            int_right = new_line.intersection(right)

            pos_intersections = []

            def check_and_add_point(point, pos_intersections):
                if point is not None and self._border_line.get_distance_to_vec(point) < 0 and self._point_in_image(
                        point):
                    pos_intersections.append(point)
                return pos_intersections

            pos_intersections = check_and_add_point(int_up, pos_intersections)
            pos_intersections = check_and_add_point(int_left, pos_intersections)
            pos_intersections = check_and_add_point(int_bottom, pos_intersections)
            pos_intersections = check_and_add_point(int_right, pos_intersections)

            if len(pos_intersections) == 0:
                raise ValueError

            if len(pos_intersections) == 1:
                return pos_intersections[0]

            else:
                max_dist = -1
                max_int = None
                for intersection in pos_intersections:
                    tmp_dist = np.linalg.norm(self._v_point - intersection)
                    if tmp_dist > max_dist:
                        max_dist = tmp_dist
                        max_int = intersection

                return max_int

        def _are_on_same_border(self, vec1, vec2):
            return (vec1[0] == 0 and vec2[0] == 0) or (
                    vec1[0] == (self.im_height) and vec2[0] == (self.im_height)) or (
                           vec1[1] == 0 and vec2[1] == 0) or (
                           vec1[1] == (self.im_width) and vec2[1] == (self.im_width))

        def _is_on_image_edge(self, vec):
            return np.array_equal(vec, np.array([0, 0])) or np.array_equal(vec, np.array(
                [self.im_height, 0])) or np.array_equal(vec, np.array([0, self.im_width])) or np.array_equal(vec,
                                                                                                             np.array([
                                                                                                                 self.im_height,
                                                                                                                 self.im_width]))

        def _get_max_dist_to_border(self):
            vec1 = self._get_image_border_intersection(self.line1)
            vec2 = self._get_image_border_intersection(self.line2)

            max_dist = max(euclidean_distance(self._v_point, vec1), euclidean_distance(self._v_point, vec2))

            vec1 = vec1.astype('int')
            vec2 = vec2.astype('int')

            if not self._are_on_same_border(vec1, vec2):
                # Find the point that is within the high confidence area
                left = vec1[1] == 0 or vec2[1] == 0
                bottom = vec1[0] == self.im_height or vec2[0] == self.im_height
                right = vec1[1] == self.im_width or vec2[1] == self.im_width
                top = vec1[0] == 0 or vec2[0] == 0

                # Exactly two conditions must be true
                conditions = np.array([left, bottom, right, top])
                assert np.sum(conditions) == 2 or (self._is_on_image_edge(vec1) or self._is_on_image_edge(vec2)), \
                    "If the two vectors are not on the same border or image edges, the number of affected borders must be 2!"

                # top_left = np.array([0, 0])
                # top_right = np.array([0, self.im_width])
                # bottom_left = np.array([self.im_height, 0])
                # bottom_right = np.array([self.im_height, self.im_width])
                #
                # comp_vecs = []
                #
                # if left and right:
                #     if self.line1.get_distance_to_vec(top_left) < 0:
                #         comp_vecs.append(bottom_left)
                #     else:
                #         comp_vecs.append(top_left)
                #
                #     i
                #
                # if left and right:
                #     comp_vecs.append(np.array([self.im_height,0]), np.array([self.im_height, self.im_width]))
                # if top and bottom

                comp_vec = np.array([0, 0])
                if left:
                    comp_vec[1] = 0
                if bottom:
                    comp_vec[0] = self.im_height
                if right:
                    comp_vec[1] = self.im_width
                if top:
                    comp_vec[0] = 0

                # # top_left_dist = self._border_line.get_distance_to_vec(np.array([0., 0.]))
                # top_left_dist = euclidean_distance(self._v_point, np.array([0., 0.]))
                # # top_right_dist = self._border_line.get_distance_to_vec(np.array([self.im_width, 0]))
                # top_right_dist = euclidean_distance(self._v_point, np.array([self.im_width, 0]))
                # # bottom_left_dist = self._border_line.get_distance_to_vec(np.array([0, self.im_height]))
                # bottom_left_dist = euclidean_distance(self._v_point, np.array([0, self.im_height]))
                # # bottom_right_dist = self._border_line.get_distance_to_vec(np.array([self.im_width, self.im_height]))
                # bottom_right_dist = euclidean_distance(self._v_point, np.array([self.im_width, self.im_height]))

                max_dist = max(max_dist, euclidean_distance(comp_vec, self._v_point))

            return max_dist

        def get_confidence_area(self, point):
            if self._border_line.get_distance_to_vec(point) > 0:
                return -1  # Abstain
            if self.line1.get_distance_to_vec(point) < 0:
                return 1  # Low 1
            if self.line2.get_distance_to_vec(point) < 0:
                return 2  # Low 2
            return 0  # High

        def _get_depth_for_lc_area(self, line, point):
            dist_to_border = self._border_line.get_distance_to_vec(point)
            dist_to_line = line.get_distance_to_vec(point)

            dist = -dist_to_line  # negated since the distance is negative in this case
            if dist > self._confidence_range:
                confidence = 0
            else:
                confidence = 1 - dist / self._confidence_range

            depth_value = (1 - dist_to_border / self._max_dist_2_border) + \
                          dist_to_border / self._max_dist_2_border * (1 - confidence)

            # TODO: Use avgvanishingpoint?
            return depth_value, confidence * confidence

        def get_depth_and_confidence(self, point):
            # Point is array of x and y coordinate
            area = self.get_confidence_area(point)
            if area == 0:  # High confidence
                dist_to_border = self._border_line.get_distance_to_vec(point)
                # dist_to_border = euclidean_distance(self._v_point, point)
                confidence = 1
                # TODO: use avgvanishingpoint?
                depth_feature = 1 - dist_to_border / self._max_dist_2_border
            elif area == 1:  # Low 1
                depth_feature, confidence = self._get_depth_for_lc_area(self.line1, point)
            elif area == 2:  # Low 2
                depth_feature, confidence = self._get_depth_for_lc_area(self.line2, point)
            else:  # Abstain
                confidence = 0
                depth_feature = 0

            return depth_feature, confidence

    def extract_features(self, segments, image):

        if len(segments) < 1:
            logging.warning("Warning: Empty input is given to feature extractor.")
            return None

        result_features = np.zeros(len(segments))

        result_confidences = []

        def is_valid_pair(height, width, line1, line2):
            # TODO: Check this, its the standard behavior in the reference implementation but seems to be wrong
            if line1.is_intersecting(line2):
                return False

            intersection = line1.intersection(line2)
            if intersection is None or np.abs(intersection[0]) > width * 5 or np.abs(intersection[1]) > height * 5:
                return False

            if line1.finite_intersection(line2) is not None and line2.finite_intersection(line1) is not None:
                return False

            return True

        origin_height = image.shape[0]
        origin_width = image.shape[1]

        logging.debug("Detecting hough lines...")
        lines = detect_hough_lines(image)

        logging.debug("Applying line fusion filter...")
        # This filter takes quite long
        lines = line_fusion_filter(lines)

        logging.debug("Applying line length filter...")
        length_threshold = min(origin_width, origin_height) / 5
        lines = line_length_filter(lines, threshold=length_threshold)

        logging.debug("Applying line angle filter...")
        angle_threshold = 10.
        lines = line_angle_filter(lines, angle_threshold)
        if self.verbose > 1:
            draw_lines_in_image(image, lines, show_image=True)

        logging.debug("Applied line filters. Keeping {} lines.".format(len(lines)))

        # Build vanishing points
        v_points = []
        raw_v_points = []
        for lin_i in range(len(lines)):
            line1 = lines[lin_i]
            for lin_j in range(lin_i + 1, len(lines)):
                if lin_i == lin_j:
                    continue

                line2 = lines[lin_j]
                if is_valid_pair(origin_height, origin_width, line1, line2):
                    new_v_point = LinearPerspectiveFeatureExtractor.VanishingPoint(line1, line2, origin_height,
                                                                                   origin_width)
                    v_points.append(new_v_point)
                    raw_v_points.append(new_v_point.get_v_point)

        clean_points = []
        for vp1 in v_points:
            if len(clean_points) == 0:
                clean_points.append(vp1)
                continue
            p1 = vp1.get_v_point
            dist = [abs(p1[0] - vp2.get_v_point[0]) + abs(p1[1] - vp2.get_v_point[1]) for vp2 in clean_points]
            if min(dist) > 50:
                clean_points.append(vp1)

        v_points = clean_points

        # Plot v points
        if self.verbose > 1:
            logging.info("Plotting points in image...")
            # plot_points_in_image(image, raw_v_points, show_image=True)
            plot_v_points(image, v_points)

        # TODO: Use AVGVanishingPoint?

        # Determine the depth value for each pixel
        if self.target_width is None:
            im_height = origin_height
        else:
            im_height = self.target_height
        if self.target_width is None:
            im_width = origin_width
        else:
            im_width = self.target_width

        depth_map = np.zeros([im_height, im_width])
        confidences = np.zeros([im_height, im_width])
        if len(v_points) == 0:
            logging.warning("No vanishing points could be found. Therefore, the depth map calculation can not be "
                            "produced.")

        logging.info("Calculating depth and confidences for each pixel of the resulting depth map...")
        with tqdm(total=(im_height * im_width)) as pbar:
            for x in range(im_height):
                for y in range(im_width):

                    depth_values = []
                    confidence_values = []

                    lookup_x = int(round(x / im_height * origin_height))
                    lookup_y = int(round(y / im_width * origin_width))

                    for vp in v_points:
                        depth_value, confidence = vp.get_depth_and_confidence(np.array([lookup_x, lookup_y]))
                        depth_values.append(depth_value)
                        confidence_values.append(confidence)

                    depth_values = np.array(depth_values)
                    confidence_values = np.array(confidence_values)
                    confidence_sum = np.sum(confidence_values)

                    final_depth = 0
                    for val_idx in range(depth_values.shape[0]):
                        final_depth += depth_values[val_idx] * (confidence_values[val_idx] / confidence_sum)
                        confidences[x, y] += confidence_values[val_idx]
                    depth_map[x, y] = final_depth
                pbar.update(im_width)
        logging.info("Finished depth and confidence calculation.")

        # Normalize depth map
        min_val = np.min(depth_map)
        depth_map = (depth_map - min_val) / (np.max(depth_map) - min_val + 0.000001)

        visualize_single_map(depth_map)

        for seg_idx in range(len(segments)):
            # Calculate features for segments
            adj_mask = mask_array_varying_size(depth_map, segments[seg_idx].get_mask)
            seg_depths = depth_map[adj_mask]
            # The normalization by the number of vanishing points has already been performed before
            depth_feature = np.sum(seg_depths) / (np.sum(adj_mask) + 0.00001)
            # TODO: Scale by maximal value?
            result_features[seg_idx] = depth_feature

            # logging.info("Finished calculation of result features {}.".format(result_features))

        confidences /= np.max(confidences)

        if self.verbose > 1:
            logging.info("Plotting confidences...")
            visualize_single_map(confidences, title="Confidences")
        result_confidences.append(confidences)

        self.result_confidences = result_confidences
        return result_features


class TargetColorSimilarity(FeatureExtractor):
    def __init__(self, target_color, hsb=False):
        assert len(target_color) == 3, "The target color must be a RGB or HSB value!"

        # Store the target saturation
        if hsb:
            self.target_saturation = target_color[1]
        else:
            self.target_saturation = color_utils.rgb_to_hsv(target_color)[1]

    def extract_features(self, segments, image):
        if len(segments) < 1:
            logging.warning("Warning: Empty input is given to feature extractor.")
            return None

        result_features = np.zeros(len(segments))

        for seg_idx in range(len(segments)):
            seg_im = segments[seg_idx].get_img_crop(image)

            seg_im_sat = get_saturation(seg_im)
            result_features[seg_idx] = np.mean(np.abs(self.target_saturation - seg_im_sat))

        return result_features
