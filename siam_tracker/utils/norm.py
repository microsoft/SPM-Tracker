# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

class ImageNormalizer(object):

    def __init__(self, mean, std, in_type='opencv', out_type='pil'):
        """
        Normalize input tensor by substracting mean value & scale std value.
        """
        self.mean = mean
        self.std = std

        assert in_type in ("opencv", "pil"), "Type must be 'opencv' or 'pil'"
        assert out_type in ("opencv", "pil"), "Type must be 'opencv' or 'pil'"

        if in_type == out_type:
            self.order_trans = False
            self.scale_factor = 1.0
        elif in_type == 'opencv' and out_type == 'pil':
            self.order_trans = True
            self.div_factor = 255.0
        elif in_type == 'pil' and out_type == 'opencv':
            self.order_trans = True
            self.div_factor = 1.0 / 255.0
        else:
            raise ValueError("Unknown key for {} {}".format(in_type, out_type))

    def __call__(self, img_tensor):

        if self.order_trans:
            img_tensor = img_tensor[:, [2, 1, 0], :, :].contiguous()
            img_tensor.div_(self.div_factor)

        for i in range(3):
            img_tensor[:, i, :, :].sub_(self.mean[i]).div_(self.std[i])

        return img_tensor
