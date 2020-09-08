import abc
import numpy as np
from mde.feature_extraction.img_utils import crop_segment
from scipy import ndimage

from mde.math_utils import normalize_vec, get_orthogonal_vec


class Line:
    def __init__(self, support_vector, direction):
        self._support_vector = np.copy(support_vector)
        self._direction = np.copy(direction)

    @staticmethod
    def build_from_points(p1, p2):
        return Line(np.copy(p1), np.copy(p2 - p1))

    @property
    def support_vector(self):
        return self._support_vector

    @property
    def direction(self):
        return self._direction

    @property
    def p1(self):
        return self._support_vector

    @property
    def p2(self):
        return self._direction + self._support_vector

    @property
    def x1(self):
        return self.p1[0]

    @property
    def y1(self):
        return self.p1[1]

    @property
    def x2(self):
        return self.p2[0]

    @property
    def y2(self):
        return self.p2[1]

    def set_support_vector(self, support_vector):
        self._support_vector = support_vector

    def set_direction(self, direction):
        self._direction = direction

    def get_orthogonal(self):
        return Line(self.support_vector, self.get_orthogonal_direction())

    def get_orthogonal_direction(self):
        return get_orthogonal_vec(self._direction)

    def invert_direction(self):
        self._direction *= -1

    # TODO: Can I speedup this? It is a little performance bottleneck
    # TODO: This is the worst name ever for this kind of function!
    def get_distance_to_vec(self, vec):
        """
        This function calculates the distance of a given vector to the line. First, it constructs a line based on the
        vector as support vector and the orthogonal direction of the line as direction. Then, the intersection is
        determined. This intersection point is "unapplied" on the line, i.e. the x step on the line is determined.
        As a result, the distance is then based on measuring where the orthogonal line is located with regard to the
        x coordinate.

        :param vec: vector to be compared to the line
        :return: // Returns the x value producing the intersection's y value for the line given as f = m*x + b
        """
        ortho_line = Line(vec, self.get_orthogonal_direction())
        intersection = self.intersection(ortho_line)
        # new_line = Line(intersection, self.get_orthogonal_direction())
        return ortho_line.unapply(intersection)
        # return euclidean_distance(intersection, vec)

    def unapply(self, vec):
        """
        Function measures the parameter on the line with regard to x producing the corresponding vector
        """
        t1, t2 = 0., 0.
        if self._direction[0] != 0. and self._direction[1] != 0.:
            t1 = (vec[0] - self._support_vector[0]) / self._direction[0]
            t2 = (vec[1] - self._support_vector[1]) / self._direction[1]
        elif self._direction[0] != 0.:
            t1 = (vec[0] - self._support_vector[0]) / self._direction[0]
            t2 = t1
        elif self._direction[1] != 0.:
            t2 = (vec[1] - self._support_vector[1]) / self._direction[1]
            t1 = t2

        if np.abs(t1 - t2) < 0.001:
            return t1

        return np.nan

    def intersection(self, other_line):
        def perp(a):
            b = np.empty_like(a)
            b[0] = -a[1]
            b[1] = a[0]
            return b

        a1 = self.p1
        a2 = self.p2
        b1 = other_line.p1
        b2 = other_line.p2

        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        result = (num / denom.astype(float)) * db + b1

        if np.isinf(result).any() or np.isnan(result).any():
            return None
        return result

    def is_intersecting(self, other_line):
        # Globally

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        a1 = self.p1
        a2 = self.p2
        b1 = other_line.p1
        b2 = other_line.p2

        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

    def finite_intersection(self, other_line):

        if not self.is_intersecting(other_line):
            return None

        # TODO: Is there a different effect compared to just used other_line as reference?
        sup = other_line.p1
        dir = other_line.p2
        dir -= sup
        f = Line(sup, dir)

        intersection = self.intersection(f)

        t = f.unapply(intersection)
        if t < 0 or t > 1:
            return None

        return intersection

    def get_bisection(self, other_line):
        """
        Function calculating a bisection between the line and another line. The resulting bisection line is a line
        "in between" both direction lines, weighted by the length of the direction.
        """

        intersection = self.intersection(other_line)
        if intersection is not None:
            new_dir = other_line.direction + self.direction
            new_dir = normalize_vec(new_dir)
            return Line(intersection, new_dir)

        return None

    def normalize_direction(self):
        length = np.sqrt(np.sum(np.square(self._direction)))
        self._direction = self._direction / length


class Segment(abc.ABC):
    def __init__(self):
        self.mask = None
        self.cropped_img = None

    @property
    def get_mask(self):
        return self.mask

    @abc.abstractmethod
    def get_center(self):
        pass

    # TODO: Analyze memory consumption
    def get_img_crop(self, image, override=False):
        if self.cropped_img is None or override:
            self.cropped_img = crop_segment(self.get_mask, image)
        return self.cropped_img

    def free_memory(self):
        if self.cropped_img is not None:
            del self.cropped_img
            self.cropped_img = None


class RectangularSegment(Segment):
    def __init__(self, start_h, end_h, start_w, end_w, origin_shape):
        super().__init__()

        self.origin_shape = origin_shape

        self.start_h = start_h
        self.end_h = end_h
        self.start_w = start_w
        self.end_w = end_w

    @property
    def get_center(self):
        """
        :return: Returns the height and width of the image's center
        """
        return (self.end_h - self.start_h) / 2 + self.start_h, (self.end_w - self.start_w) / 2 + self.start_w

    @property
    def get_start_h(self):
        return self.start_h

    @property
    def get_end_h(self):
        return self.end_h

    @property
    def get_start_w(self):
        return self.start_w

    @property
    def get_end_w(self):
        return self.end_w

    @property
    def get_mask(self):
        tmp_mask = np.zeros(self.origin_shape)
        tmp_mask[self.start_h: self.end_h, self.start_w: self.end_w] = 1
        return tmp_mask.astype(bool)


class IrregularSegment(Segment):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    @property
    def get_center(self):
        return ndimage.measurements.center_of_mass(self.mask)
