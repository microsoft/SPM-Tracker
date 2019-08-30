# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torch.autograd import Function
from . import crop_and_resize as _B


class CropAndResizeFunction(Function):

    def __init__(self, crop_height, crop_width, extrapolation_value=0.0, has_normed=False):
        """ Torch Function for crop and resize (bilinear) implementation
        Args:
            crop_height: the output height, int
            crop_width: the output width, int
            extrapolation_value: paddding value for some pixel outside the source input, 0.0 by default
            has_normed: if the input coordinate is normalized by image size (height & width).
        """
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value
        self.has_normed = has_normed

    def forward(self, image, boxes, box_inds):
        """
        Args:
            image: torch.Tensor of shape [N, C, H, W], 4 dimention is supported only.
            boxes: torch.Tensor of shape [K, 4], the cropping box coordinates, in order of [x1, y1, x2, y2]
            box_inds: torch.Tensor of shape [K, 1], the index indicates which image to crop.
        """
        box_inds = box_inds.contiguous()
        crop_boxes = boxes[:, :4].contiguous()

        n, c, h, w = image.size()
        self.batch_size = int(n)
        self.image_height = int(h)
        self.image_width = int(w)

        if not self.has_normed:
            # normalization if necessary
            crop_boxes[:, 0:4:2] /= (w - 1)
            crop_boxes[:, 1:4:2] /= (h - 1)

        if image.is_cuda:
            # if image is on GPU device. We should crop it in the same device.
            with torch.cuda.device(image.device.index):
                crops = _B.forward(image, crop_boxes, box_inds,
                                   self.extrapolation_value, self.crop_height, self.crop_width)
        else:
            crops = _B.forward(image, crop_boxes, box_inds,
                               self.extrapolation_value, self.crop_height, self.crop_width)

        # save for backward
        self.save_for_backward(crop_boxes, box_inds)

        return crops

    def backward(self, grad_outputs):
        boxes, box_ind = self.saved_tensors
        grad_outputs = grad_outputs.contiguous()
        if grad_outputs.is_cuda:
            with torch.cuda.device(grad_outputs.device.index):
                grads_image = _B.backward(grad_outputs, boxes, box_ind,
                                          self.batch_size, self.image_height, self.image_width)
        else:
            grads_image = _B.backward(grad_outputs, boxes, box_ind,
                                      self.batch_size, self.image_height, self.image_width)
        return grads_image, None, None
