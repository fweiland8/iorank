import abc
import numpy as np

from mde.feature_extraction.shapes import RectangularSegment


class SegmentExtractor(abc.ABC):
    @abc.abstractmethod
    def extract_segments(self, dataset):
        pass

    @staticmethod
    def _retrieve_img_shape(dataset, idx):
        if dataset.get_img_shape() is None:
            im = dataset._get_image(idx)
            im_shape = np.copy(im.shape)
            del im
        else:
            im_shape = dataset.get_img_shape()
        return im_shape


class UniformSegmentExtractor(SegmentExtractor):
    def __init__(self, num_segs_per_axis):
        self.num_segs_per_axis = num_segs_per_axis

    def extract_segments(self, dataset):
        final_segments = []

        # Assumes images shape of (height, width, [channels])
        for idx in range(dataset.get_num_instances()):
            im_shape = SegmentExtractor._retrieve_img_shape(dataset, idx)

            segments = []
            seg_shape = (np.array(im_shape[:2]) // self.num_segs_per_axis).astype(int)
            seg_shape_rest = (np.array(im_shape[:2]) % self.num_segs_per_axis).astype(int)

            for h_step in range(self.num_segs_per_axis):
                for w_step in range(self.num_segs_per_axis):
                    h_start = h_step * seg_shape[0]
                    h_end = (h_step + 1) * seg_shape[0]
                    if h_step == self.num_segs_per_axis - 1:
                        h_end += seg_shape_rest[0]

                    w_start = w_step * seg_shape[1]
                    w_end = (w_step + 1) * seg_shape[1]
                    if w_step == self.num_segs_per_axis - 1:
                        w_end += seg_shape_rest[1]

                    # Outputs boolean masks as segments
                    seg = RectangularSegment(h_start, h_end, w_start, w_end, im_shape[:2])
                    segments.append(seg)
            final_segments.append(segments)

        return final_segments


class SuperpixelSegmentExtractor(SegmentExtractor):
    def __init__(self, num_segments_width, num_segments_height):
        self.num_segments_width = num_segments_width
        self.num_segments_height = num_segments_height

    def _calculate_steps(self, im_shape):
        h_step = im_shape[0] // self.num_segments_height
        h_step_rest = im_shape[0] % self.num_segments_height
        w_step = im_shape[1] // self.num_segments_width
        w_step_rest = im_shape[1] % self.num_segments_width

        return h_step, h_step_rest, w_step, w_step_rest

    def extract_segments(self, dataset):
        final_segments = []

        for idx in range(dataset.get_num_instances()):
            segments = []

            im_shape = SegmentExtractor._retrieve_img_shape(dataset, idx)

            h_step, h_step_rest, w_step, w_step_rest = self._calculate_steps(im_shape)

            for h_idx in range(self.num_segments_height):
                for w_idx in range(self.num_segments_width):
                    tmp_h_step = h_step
                    if h_idx == self.num_segments_height - 1:
                        tmp_h_step += h_step_rest

                    tmp_w_step = w_step
                    if w_idx == self.num_segments_width - 1:
                        tmp_w_step += w_step_rest

                    seg = RectangularSegment(h_idx * h_step, h_idx * h_step + tmp_h_step, w_idx * w_step,
                                             w_idx * w_step + tmp_w_step, im_shape)
                    segments.append(seg)

            final_segments.append(segments)
        return final_segments
