import logging
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from iorank.util.util import get_device, is_dummy_box, blacken_image, get_patch_from_image


class VisualAppearanceFeatureTransformer:
    def __init__(self, input_type='default', reduced_size=64, use_masks=False, **kwargs):
        """
        Parent class for visual appearance feature transformers.

        :param input_type: Input type for the auto encoder. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param use_masks: If True, object masks are applied to the image patches before they are processed. Default: False
        :param kwargs: Keyword arguments
        """
        self.logger = logging.getLogger(VisualAppearanceFeatureTransformer.__name__)

        self.device = get_device()
        self.use_masks = use_masks
        self.input_type = input_type

        # Tunable parameters
        self.reduced_size = reduced_size

    def transform(self, rgb_images, all_boxes, all_labels, all_masks=None):
        """
        Produces for each of the provided images and each object a feature vector. The feature vector is a
        flattened representation of a spatial mask for a bounding box.

        N : Batch size \n
        U: Padding size (upper bound for the number of objects) \n
        H: Image height \n
        W: Image width \n
        F: Number of produced features

        :param rgb_images: Tensor of RGB images of size (N,3,H,W)
        :param all_boxes: Tensor of bounding box coordinates of size (N,U,4)
        :param all_labels: Tensor of class labels of size (N,U)
        :param all_masks: Optional: Tensor of object masks of size (N,U,H,W)
        :return: Feature matrix of size (N,U,F)
        """
        if self.use_masks and all_masks is None:
            raise RuntimeError("No masks are provided but transformer has use_masks = True")

        n_instances = all_boxes.size()[0]
        n_objects = all_boxes.size()[1]

        # Get object patches dependent on the input type
        if self.input_type == 'default':
            inputs = self.create_model_input_default(rgb_images, all_boxes, all_masks)
        elif self.input_type == 'blacked':
            inputs = self.create_model_input_blacked(rgb_images, all_boxes, all_masks)
        else:
            raise RuntimeError("Invalid input type: {}".format(self.input_type))
        inputs = inputs.to(self.device, non_blocking=True)

        n_inputs = inputs.size()[0]

        # How the features are obtained for an input (= a patch) depends on the concrete feature transformer
        features = self.get_features_for_input(inputs)

        # Add dummy bit
        features = features.view(n_inputs, -1)
        features = F.pad(features, (0, 1), mode='constant', value=0)

        # Build result tensor
        ret = torch.ones(size=(n_instances, n_objects, self.get_n_features()), device=self.device)
        idx = 0
        for i in range(n_instances):
            n_instance_objects = torch.sum(torch.all(torch.ne(all_boxes[i], -1), dim=1).float())
            for j in range(int(n_instance_objects)):
                ret[i][j] = features[idx]
                idx += 1

        return ret

    def create_model_input_blacked(self, rgb_images, all_boxes, all_masks):
        """
        Creates the model input (i.e. the intermediate patches) in the 'blacken' strategy.

        N : Batch size \n
        U: Padding size (upper bound for the number of objects) \n
        H: Image height \n
        W: Image width

        :param rgb_images: Tensor of RGB images of size (N,3,H,W)
        :param all_boxes: Tensor of bounding box coordinates of size (N,U,4)
        :param all_masks: Optional: Tensor of object masks of size (N,U,H,W)
        :return: Tensor of intermediate patches
        """
        images = []
        for i in range(all_boxes.size(0)):
            for j in range(all_boxes.size(1)):
                box = all_boxes[i][j]
                if is_dummy_box(box):
                    continue

                image = rgb_images[i].cpu()

                # Apply object mask to the image
                if self.use_masks:
                    mask = all_masks[i][j].cpu()
                    image = image * mask

                image = TF.to_pil_image(image)
                image = blacken_image(image, box)
                image = TF.resize(image, (self.reduced_size, self.reduced_size))
                image = TF.to_tensor(image)
                images.append(image)
        return torch.stack(images)

    def create_model_input_default(self, rgb_images, all_boxes, all_masks):
        """
        Creates the model input (i.e. the intermediate patches) in the 'default'/'cut-out' strategy.

        N : Batch size \n
        U: Padding size (upper bound for the number of objects) \n
        H: Image height \n
        W: Image width

        :param rgb_images: Tensor of RGB images of size (N,3,H,W)
        :param all_boxes: Tensor of bounding box coordinates of size (N,U,4)
        :param all_masks: Optional: Tensor of object masks of size (N,U,H,W)
        :return: Tensor of intermediate patches
        """
        box_regions = []
        for i in range(all_boxes.size(0)):
            for j in range(all_boxes.size(1)):
                box = all_boxes[i][j]
                if is_dummy_box(box):
                    continue
                image = rgb_images[i].cpu()

                # Apply object mask to the image
                if self.use_masks:
                    image = image.clone()
                    mask = all_masks[i][j].cpu()
                    image = image * mask

                box_region = get_patch_from_image(box, image)
                box_regions.append(box_region)

        t = T.Compose([T.ToPILImage(), T.Resize((self.reduced_size, self.reduced_size)), T.ToTensor()])
        box_regions = [t(box.cpu()) for box in box_regions]
        return torch.stack(box_regions)

    def set_tunable_parameters(self, input_type='default', reduced_size=64, **kwargs):
        """
        Sets tunable parameters for this model.

        :param input_type: Input type for the auto encoder. If 'default', then an object region is cut out.
            If 'blacked', then the image is blacked except for the object region. Default: 'default'
        :param reduced_size: Size to which inputs are scaled before being processed. Default: 64
        :param kwargs: Keyword arguments
        """
        self.logger.info("Set parameters: input_type=%s, reduced_size=%s", input_type, reduced_size)
        self.input_type = input_type
        self.reduced_size = reduced_size
