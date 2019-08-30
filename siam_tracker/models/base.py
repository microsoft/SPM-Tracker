# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch.nn


class NetworkInfo(object):

    __slots__ = ('stride', 'channel', 'rf', 'size_func')

    def __init__(self, stride, channel, rf, size_func):
        """ Network information in echo stage.
        Args:
            stride (int): feature stride in the target stage
            channel (int): the number of channels in the target stage
            rf (int): receptive field in the target stage
            size_func (callable): function, given the input size, calculate
                                  the output size in the target stage (if padding).
        """
        self.stride = stride
        self.channel = channel
        self.rf = rf
        self.size_func = size_func


class BackBoneCNN(torch.nn.Module):
    """ Basic class for network backbone. """

    # static attribute
    name = "base"
    num_blocks = 0
    blocks = dict()

    @classmethod
    def infer_size(cls, tensor_name, in_size, padding=None):
        """ Given output tensor name and input size, infer the output size """
        if padding:
            return cls._infer_size_with_padding(tensor_name, in_size)
        else:
            return cls._infer_size_wo_padding(tensor_name, in_size)

    @classmethod
    def _infer_size_wo_padding(cls, tensor_name, in_size):
        _tensor_name_list = tensor_name if isinstance(tensor_name, (list, tuple)) else [tensor_name]
        out_size_list = []
        for _tensor_name in _tensor_name_list:
            _info = cls.blocks[_tensor_name]
            out_size_list.append(int((in_size - _info.rf) / _info.stride) + 1)
        if isinstance(tensor_name, (list, tuple)):
            return out_size_list
        else:
            return out_size_list[0]

    @classmethod
    def _infer_size_with_padding(cls, tensor_name, in_size):
        _tensor_name_list = tensor_name if isinstance(tensor_name, (list, tuple)) else [tensor_name]
        out_size_list = []
        for _tensor_name in _tensor_name_list:
            _info = cls.blocks[_tensor_name]
            out_size_list.append(_info.size_func(in_size))
        if isinstance(tensor_name, (list, tuple)):
            return out_size_list
        else:
            return out_size_list[0]

    @classmethod
    def infer_channels(cls, tensor_name):
        """ Given output tensor name, return the number of channels.
        If tensor name is a string, it will return a integer.
        If tensor name is a list or tuple, it will return a list of integer.
        """
        _tensor_name_list = tensor_name if isinstance(tensor_name, (list, tuple)) else [tensor_name]
        out_list = []
        for _tensor_name in _tensor_name_list:
            _info = cls.blocks[_tensor_name]
            out_list.append(_info.channel)
        if isinstance(tensor_name, (list, tuple)):
            return out_list
        else:
            return out_list[0]

    @classmethod
    def infer_strides(cls, tensor_name):
        """ Given output tensor name, return the number of channels.
            If tensor name is a string, it will return a integer.
            If tensor name is a list or tuple, it will return a list of integer.
        """
        _tensor_name_list = tensor_name if isinstance(tensor_name, (list, tuple)) else [tensor_name]
        out_list = []
        for _tensor_name in _tensor_name_list:
            _info = cls.blocks[_tensor_name]
            out_list.append(_info.stride)
        if isinstance(tensor_name, (list, tuple)):
            return out_list
        else:
            return out_list[0]

    def __init__(self):
        super(BackBoneCNN, self).__init__()

    def freeze_block(self, num_blocks):
        """ Freeze the parameters in the target blocks """
        for i in range(num_blocks):
            sub_module = getattr(self, 'conv{}'.format(i+1))
            for m in sub_module.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False
                elif isinstance(m, torch.nn.BatchNorm2d):
                    if m.weight is not None:
                        m.weight.requires_grad = False
                    if m.bias is not None:
                        m.bias.requires_grad = False
                    m.eval()

    def forward(self, x):
        out = dict()
        for i in range(1, self.num_blocks+1):
            op_name = 'conv{}'.format(i)
            op = getattr(self, op_name)
            if i == 1:
                out['conv1'] = op(x)
            else:
                out[op_name] = op(out['conv{}'.format(i-1)])
        return out
