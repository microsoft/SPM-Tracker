# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image

from .config import cfg
from .tracker_info import tracker_info_dict
from ..utils.norm import ImageNormalizer


class SiamTracker(object):
    """ A class for Siamese network based visual tracking algorithms, including SiamFC,
    SiamRPN, SA-Siam and SPM-Tracker.
    """

    def __init__(self, model_device):

        tracker_type = cfg.TRACKER_TYPE.lower()

        if tracker_type not in tracker_info_dict:
            print("Error, unknown tracker type {}, please check cfg.TRACKER_TYPE".format(tracker_type))
            raise ValueError
        tracker_info = tracker_info_dict[tracker_type]

        # construct the image normailizer
        if cfg.USE_PIL_LIB:
            img_norm = ImageNormalizer(cfg.PIL_MEAN, cfg.PIL_STD, in_type='pil', out_type='pil')
        else:
            img_norm = ImageNormalizer(cfg.CV_MEAN, cfg.CV_STD, in_type='opencv', out_type='opencv')

        self.tracker = tracker_info['inference_wrapper'](model_device, img_norm, cfg)
        self.model_device = model_device

    def track(self, init_box, input_str, box_format='tlbr'):
        """ Track API, depend on the input (image path list or video path), the function
        will return the track results.

        Args:
            init_box: 4 values that indicate the object postion.
            input_str: image path list or video path (not supported yet)
            box_format: formats of box coordinate.

        Returns:
            tracked_box_list [x1, y1, x2, y2]

        """

        if box_format == 'tlbr':
            box = init_box
        elif box_format == 'xywh':
            raise NotImplementedError
        elif box_format == 'tlwh':
            box = [(init_box[0]),
                   (init_box[1]),
                   (init_box[2] + init_box[0]),
                   (init_box[3] + init_box[1])]
        else:
            print("Unsupported type of box format: {}".format(box_format))
            raise NotImplementedError

        if isinstance(input_str, list):
            results = self.track_image_list(box, input_str)
        else:
            # TODO(guangting): add video support
            raise NotImplementedError

        return results

    def track_image_list(self, init_box, image_list):
        """ Given a list of image input, track the object in initialization postion

        Args:
            image_list: image path list, a list of str.
            init_box: initialization position in order of [x1, y1, x2, y2]

        Returns:
            track_box_list: a list that consists of tracking boxes.
        """

        current_box = init_box
        tracked_box_list = []

        # init the sequence list & data loader
        seq_dt = SequenceLoader(image_list, use_pil=cfg.USE_PIL_LIB)
        seq_loader = DataLoader(seq_dt, batch_size=1, shuffle=False, sampler=None, num_workers=1, pin_memory=True)

        for image_idx, img in enumerate(seq_loader):
            img = img.to(self.model_device)
            vis = cfg.INFERENCE.DEBUG_VIS_ON and image_idx % cfg.INFERENCE.DEBUG_VIS_STEP == 0
            vis = vis and image_idx > cfg.INFERENCE.DEBUG_VIS_START_FRAME
            if image_idx == 0:
                self.tracker.init_tracker(img, current_box)
            else:
                current_box = self.tracker.predict_next_frame(img, current_box, vis)
            tracked_box_list.append(current_box[0:4])

        return tracked_box_list


class SequenceLoader(Dataset):
    """ A dataset for loading images from a sequence. """

    def __init__(self, image_list, use_pil=False):

        self.image_list = image_list
        self.use_pil = use_pil

        if use_pil:
            self.to_tensor = ToTensor()

    def __getitem__(self, idx):

        # load images in PIL format
        # img = Image.open()
        if self.use_pil:
            img = Image.open(self.image_list[idx])
            if img.mode == 'L':
                img = img.convert('RGB')  # convert to RGB 3 channels if necessary
            img_tensor = self.to_tensor(img)
        else:
            img = cv2.imread(self.image_list[idx], cv2.IMREAD_COLOR).astype(np.float32)
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))

        return img_tensor

    def __len__(self):
        return len(self.image_list)
