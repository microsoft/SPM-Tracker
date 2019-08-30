# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Crop operations"""
import torch
import numpy as np

from ..ops.crop_and_resize import CropAndResizeFunction

def center_crop_tensor(t, crop_height, crop_width):
    """ Given the output size (crop_height, crop_width), center crop the target tensor """

    n, c, h, w = t.size()
    if h == crop_height and w == crop_width:
        return t
    sy = (h - crop_height) // 2
    ty = sy + crop_height
    sx = (w - crop_width) // 2
    tx = sx + crop_width
    crop_tensor = t[:, :, sy:ty, sx:tx].contiguous()
    return crop_tensor


def crop_with_boxes(img_tensor, x_crop_boxes, out_height, out_width, crop_inds=None, avg_channels=True,
                    has_normed_coords=False):
    """Crop the image tensor by given boxes. The output will be resized to target size

    Params:
        img_tensor: torch.Tensor, in shape of [N, C, H, W]. If N > 1, the crop_inds must be specified.
        crop_boxes: list/numpy.ndarray/torch.Tensor in shape of [K x 4].
        out_height: int.
        out_width: int.
        crop_inds: list/numpy.ndarray/torch.Tensor in shape of [K]
    Returns:
        crop_img_tensor: torch.Tensor, in shape of [K, C, H, W]
    """

    img_device = img_tensor.device

    if isinstance(x_crop_boxes, list):
        crop_boxes = torch.tensor(x_crop_boxes, dtype=torch.float32).to(img_device)
    elif isinstance(x_crop_boxes, np.ndarray):
        crop_boxes = torch.tensor(x_crop_boxes, dtype=torch.float32).to(img_device)
    elif isinstance(x_crop_boxes, torch.Tensor):
        # change type and device if necessary
        crop_boxes = x_crop_boxes.clone().to(device=img_device, dtype=torch.float32)
    else:
        raise ValueError('Unknown type for crop_boxes {}'.format(type(x_crop_boxes)))

    if len(crop_boxes.size()) == 1:
        crop_boxes = crop_boxes.view(1, 4)

    num_imgs, chanenls, img_height, img_width = img_tensor.size()
    num_crops = crop_boxes.size(0)

    if crop_inds is not None:
        if isinstance(crop_inds, list) or isinstance(crop_inds, np.ndarray):
            crop_inds = torch.tensor(crop_inds, dtype=torch.float32).to(img_device)
        elif isinstance(crop_inds, torch.Tensor):
            crop_inds = crop_inds.to(device=img_device, dtype=torch.float32)
        else:
            raise ValueError('Unknown type for crop_inds {}'.format(type(crop_inds)))
        crop_inds = crop_inds.view(-1)
        assert crop_inds.size(0) == crop_boxes.size(0)
    else:
        if num_imgs == 1:
            crop_inds = torch.zeros(num_crops, dtype=torch.float32, device=img_device)
        elif num_imgs == num_crops:
            crop_inds = torch.arange(num_crops, dtype=torch.float32, device=img_device)
        else:
            raise ValueError('crop_inds MUST NOT be None.')

    if avg_channels:
        img_channel_avg = img_tensor.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        img_tensor_minus_avg = img_tensor - img_channel_avg  # minus mean values
    else:
        img_tensor_minus_avg = img_tensor
    crop_img_tensor = CropAndResizeFunction(out_height, out_width, has_normed=has_normed_coords)(
        img_tensor_minus_avg, crop_boxes, crop_inds)

    if avg_channels:
        # add mean value
        crop_img_tensor += img_channel_avg[crop_inds.long()]

    return crop_img_tensor
