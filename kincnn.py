"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from utils import (Conv2dNonPadding, Conv2dStaticSamePadding,
                   MemoryEfficientSwish, Swish, calculate_output_image_size,
                   drop_connect, efficientnet_params, get_model_params,
                   get_same_padding_conv2d, load_pretrained_weights,
                   round_filters, round_repeats)

# Author: HongbiKim (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = partial(Conv2dStaticSamePadding, image_size=image_size)

        # Stem
        in_channels = 1  # rgb # 이 부분 수정했음
        out_channels = round_filters(
            3, self._global_params
        )  # number of output channels
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=(15, 1), stride=(2, 1), bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )
        image_size = calculate_output_image_size(image_size, stride=2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                # input_filters=8,
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConvBlock(block_args, self._global_params, image_size=image_size)
            )
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self._global_params, image_size=image_size)
                )

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(3, self._global_params)
        # out_channels = round_filters(1280, self._global_params) <-- 원본
        # out_channels = 1280
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        # Final linear layer
        # self._avg_pooling = nn.AdaptiveAvgPool2d(1) <-- 원본
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(self.define_last_fcn(), self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints["reduction_{}".format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x

        return endpoints

    def define_last_fcn(self):
        global allele
        print(allele)
        if "phospho-A" in allele:
            return 132096
        elif "phospho-B" in allele:
            return 8448
        elif "HLA-B" in allele:
            return 11392
        elif "HLA-C" in allele:
            return 6760

    def extract_features(self, inputs):
        """Returns output of the final convolution layer"""

        # Stem
        x = self._conv_stem(inputs)
        x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        return x

    def forward(self, inputs):
        """Calls extract_features to extract features, applies final linear layer, and returns logits."""
        # bs = inputs.size(0)
        # print(bs)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        # x = x.view(bs, -1)
        # print(x.shape)

        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        # print(x.shape)

        return torch.sigmoid(x)

    @classmethod
    def from_name(cls, model_name, override_params=None):
        # cls._check_model_name_is_valid(model_name)
        global allele
        allele = model_name
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name."""
        valid_models = ["efficientnet-b" + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))
