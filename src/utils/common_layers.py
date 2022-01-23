import torch
from torch import nn

def INReLU(num_features):
    """Shorthand for InstaceNorm + Relu

    Returns:
        [type]: [description]
    """
    return torch.nn.Sequential(
        nn.InstanceNorm2d(num_features, eps=1e-5, affine=True),
        nn.ReLU()
    )


def BNReLU(num_features):
    """Shorthand for InstaceNorm + Relu

    Returns:
        [type]: [description]
    """
    return torch.nn.Sequential(
        nn.BatchNorm2d(num_features, eps=1e-5, affine=True),
        nn.ReLU()
    )
    
# -----------------------------------------------------------------------
# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
#
# Official Implementation of the CVPR2020 Paper
# Two-shot Spatially-varying BRDF and Shape Estimation
# Mark Boss, Varun Jampani, Kihwan Kim, Hendrik P. A. Lensch, Jan Kautz
# -----------------------------------------------------------------------

from typing import Callable, Optional, Tuple, Union

# import tensorflow as tf
# from tensorpack.models import (
#     BatchNorm,
#     BNReLU,
#     Conv2D,
#     Conv2DTranspose,
#     Dropout,
#     FullyConnected,
#     InstanceNorm,
#     MaxPooling,
#     layer_register,
# )
# from tensorpack.tfutils.argscope import argscope, get_arg_scope

EPS = 1e-7


def apply_mask(
    img: torch.Tensor, mask: torch.Tensor, name: str, undefined: float = 0
) -> torch.Tensor:
    return torch.where(
        torch.less_equal(mask, 1e-5), torch.ones_like(img) * undefined, img
    )


def div_no_nan(x, y):
    if y == 0:
        return torch.zeros_like(x)
    else:
        return x / y


def uncompressDepth(
    d: torch.Tensor, sigma: float = 2.5, epsilon: float = 0.7
) -> torch.Tensor:
    """From 0-1 values to full depth range. The possible depth range
        is modelled by sigma and epsilon and with sigma=2.5 and epsilon=0.7
        it is between 0.17 and 1.4.
        """
    return torch.div(1.0, 2.0 * sigma * d + epsilon)


def saturate(x, l=0.0, h=1.0):
    return torch.clip(x, l, h)


def mix(x, y, a):
    return x * (1 - a) + y * a


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.greater_equal(x, 0.04045),
        torch.pow(torch.div(x + 0.055, 1.055), 2.4),
        torch.div(x, 12.92),
    )


def linear_to_gamma(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return torch.pow(x, 1.0 / gamma)


def gamma_to_linear(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    return torch.pow(x, gamma)


def isclose(x: torch.Tensor, val: float, threshold: float = EPS) -> torch.Tensor:
    return torch.less_equal(torch.abs(x - val), threshold)


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * EPS, x))
    return torch.sqrt(sqrt_in)


def magnitude(x: torch.Tensor, data_format: str = "channels_last") -> torch.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return safe_sqrt(
        dot(x, x, data_format)
    )  # Relu seems strange but we're just clipping 0 values


def own_div_no_nan(
    x: torch.Tensor, y: torch.Tensor, data_format: str = "channels_last"
) -> torch.Tensor:
    return torch.where(torch.less(to_vec3(y, data_format), 1e-7), torch.zeros_like(x), x / y)


def normalize(x: torch.Tensor, data_format: str = "channels_last") -> torch.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return own_div_no_nan(x, magnitude(x, data_format), data_format)


def dot(x: torch.Tensor, y: torch.Tensor, data_format: str = "channels_last") -> torch.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return torch.sum(x * y, dim=get_channel_axis(data_format), keepdims=True)


def to_vec3(x: torch.Tensor, data_format: str = "channels_last") -> torch.Tensor:
    assert data_format in ["channels_last", "channels_first"]
    return repeat(x, 3, get_channel_axis(data_format))


def get_channel_axis(data_format: str = "channels_last") -> int:
    assert data_format in ["channels_last", "channels_first"]
    if data_format == "channels_first":
        channel_axis = 1
    else:
        channel_axis = -1

    return channel_axis


def repeat(x: torch.Tensor, n: int, axis: int) -> torch.Tensor:
    repeat = [1 for _ in range(len(x.shape))]
    repeat[axis] = n
    
    return torch.tile(x, repeat)

def upsample(x, factor: int = 2):
    _, h, w, _ = x.get_shape().as_list()
    x = torch.image.resize_nearest_neighbor(
        x, [factor * h, factor * w], align_corners=True
    )
    return x

# renamedto MergeConv in script merge_conv.py
# def Fusion2DBlock(
#     prevIn: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
#     filters: int,
#     kernel_size: int,
#     stride: int,
#     downscale: bool = True,
#     activation=INReLU,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     lmain = Conv2D("main_conv", prevIn[0], filters, kernel_size, activation=activation)
#     laux = Conv2D("aux_conv", prevIn[1], filters, kernel_size, activation=activation)

#     mixInput = [lmain, laux]
#     prevMixOutput = prevIn[2]
#     if prevMixOutput is not None:
#         mixInput.append(prevMixOutput)

#     mixIn = torch.concat(mixInput, -1, "mix_input")
#     lmix = Conv2D("mix_conv", mixIn, filters, kernel_size, activation=activation)

#     lmix = tf.add_n([laux, lmain, lmix], "mix_summation")

#     if stride > 1:
#         if downscale:
#             lmain = MaxPooling("main_pool", lmain, 3, strides=stride, padding="SAME")
#             laux = MaxPooling("aux_pool", laux, 3, strides=stride, padding="SAME")
#             lmix = MaxPooling("mix_pool", lmix, 3, strides=stride, padding="SAME")
#         else:
#             lmain = upsample("main_upsample", lmain, factor=stride)
#             laux = upsample("aux_upsample", laux, factor=stride)
#             lmix = upsample("mix_upsample", lmix, factor=stride)

#     return (lmain, laux, lmix)


# def resnet_shortcut(
#     l: tf.Tensor, n_out: int, stride: int, isDownsampling: bool, activation=tf.identity
# ):
#     data_format = get_arg_scope()["Conv2D"]["data_format"]
#     n_in = l.get_shape().as_list()[
#         1 if data_format in ["NCHW", "channels_first"] else 3
#     ]
#     if n_in != n_out or stride != 1:  # change dimension when channel is not the same
#         if isDownsampling:
#             return Conv2D(
#                 "convshortcut", l, n_out, 1, strides=stride, activation=activation
#             )
#         else:
#             return Conv2DTranspose(
#                 "convshortcut", l, n_out, 1, strides=stride, activation=activation
#             )
#     else:
#         return l


def resnet_shortcut(
        n_in: int, n_out: int, stride: int, isDownsampling: bool, activation=torch.nn.Identity
):
    #data_format = get_arg_scope()["Conv2D"]["data_format"]
    #n_in = l.shape.as_list()[
    #     1 if data_format in ["NCHW", "channels_first"] else 3
    # ]
    if n_in != n_out or stride != 1:  # change dimension when channel is not the same
        if isDownsampling:
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                n_in, n_out, kernel_size=1, stride=stride, padding=0),  # , bias=False),
                activation(n_out)
            )
        else:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    n_in, n_out, kernel_size=1, stride=stride, padding=0),  # , bias=False),
                activation(n_out)
            )
    else:
        return torch.nn.Identity()


def apply_preactivation(channels, preact: str):
    if preact == "bnrelu":
        layer = BNReLU(channels)
    elif preact == "inrelu":
        layer = INReLU(channels)
    else:
        layer = torch.nn.Identity(channels)
    return layer


# def preresnet_basicblock(
#     l: tf.Tensor,
#     ch_out: int,
#     stride: int,
#     preact: str,
#     isDownsampling: bool,
#     dilation: int = 1,
#     withDropout: bool = False,
# ):
#     l, shortcut = apply_preactivation("p1", l, preact)

#     if isDownsampling:
#         l = Conv2D("conv1", l, ch_out, 3, strides=stride, dilation_rate=dilation)
#     else:
#         l = Conv2DTranspose("tconv1", l, ch_out, 3, stride=stride)

#     if withDropout:
#         l = Dropout(l)
#     l, _ = apply_preactivation("p2", l, preact)

#     l = Conv2D("conv2", l, ch_out, 3, dilation_rate=dilation)

#     return l + resnet_shortcut(shortcut, ch_out, stride, isDownsampling)


def preresnet_basicblock(
    ch_in: int,
    ch_out: int,
    stride: int,
    preact: str,
    isDownsampling: bool,
    dilation: int = 1,
    withDropout: bool = False,
 ):
    layers = []
    layers.append(apply_preactivation(ch_in, preact))

    if isDownsampling:
        # TODO: test if padding is right
        layers.append(torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=dilation, dilation=dilation))
    else:
        # TODO: test if padding is right
        layers.append(torch.nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=dilation,
                                               dilation=dilation))

    if withDropout:
        layers.append(torch.nn.Dropout(ch_out))
    layers.append(apply_preactivation(ch_out, preact))

    layers.append(torch.nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=dilation, dilation=dilation))

    layers.append(resnet_shortcut(ch_in, ch_out, stride, isDownsampling))

    return torch.nn.Sequential(*layers)

# def preresnet_group(
#     name: str,
#     l: tf.Tensor,
#     block_func: Callable[[tf.Tensor, int, int, str, bool, int, bool], tf.Tensor],
#     features: int,
#     count: int,
#     stride: int,
#     isDownsampling: bool,
#     activation_function: str = "inrelu",
#     dilation: int = 1,
#     withDropout: bool = False,
#     addLongSkip: Optional[Tuple[int, tf.Tensor]] = None,
#     getLongSkipFrom: Optional[int] = None,
# ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
#     if addLongSkip and getLongSkipFrom:
#         assert addLongSkip[0] != getLongSkipFrom

#     if stride != 1:
#         assert dilation == 1
#     if dilation != 1:
#         assert stride == 1

#     with tf.variable_scope(name):
#         if addLongSkip is not None:
#             addSkipAt, skipConn = addLongSkip
#         else:
#             addSkipAt, skipConn = -1, None

#         for i in range(0, count):
#             with tf.variable_scope("block%d" % i):

#                 # first block doesn't need activation
#                 l = block_func(
#                     l,
#                     features,
#                     stride if i == 0 else 1,
#                     "no_preact" if i == 0 else activation_function,
#                     isDownsampling if i == 0 else True,
#                     dilation,
#                     withDropout,
#                 )
#                 if getLongSkipFrom is not None:
#                     if i == getLongSkipFrom:
#                         skipConnection = l

#                 if i == addSkipAt:
#                     with tf.variable_scope("long_shortcut"):
#                         changed_shortcut = resnet_shortcut(
#                             skipConn, l.shape[-1], 1, True
#                         )
#                         l = l + changed_shortcut

#         # end of each group need an extra activation
#         if activation_function == "bnrelu":
#             l = BNReLU("bnlast", l)
#         if activation_function == "inrelu":
#             l = INReLU(l)

#     if getLongSkipFrom is not None:
#         return l, skipConnection
#     else:
#         return l


def preresnet_group(
    name: str,
    ch_in: int,
    block_func: Callable[[int, int, int, str, bool, int, bool], torch.nn.Module],
    features: int,
    count: int,
    stride: int,
    isDownsampling: bool,
    activation_function: str = "inrelu",
    dilation: int = 1,
    withDropout: bool = False,
    addLongSkip = None, # : Optional[Tuple[int, tf.Tensor]] = None, TODO: add TypeHint
    getLongSkipFrom: Optional[int] = None,
) -> Tuple[torch.nn.Module]:
# ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    if addLongSkip and getLongSkipFrom:
        assert addLongSkip[0] != getLongSkipFrom

    if stride != 1:
        assert dilation == 1
    if dilation != 1:
        assert stride == 1

    layers = []
    torch_layers = []

    if addLongSkip is not None:
        addSkipAt, skipConn = addLongSkip
    else:
        addSkipAt, skipConn = -1, None

    for i in range(0, count):
        # first block doesn't need activation
        layer = block_func(
            ch_in,
            features,
            stride if i == 0 else 1,
            "no_preact" if i == 0 else activation_function,
            isDownsampling if i == 0 else True,
            dilation,
            withDropout,
        )
        layers.append(layer)

        if getLongSkipFrom is not None:
            if i == getLongSkipFrom:
                skipConnection = layer
                torch_layers.append((torch.nn.Sequential(*layers), "save"))
                layers.clear()

        if i == addSkipAt:
            #with tf.variable_scope("long_shortcut"):
            changed_shortcut = resnet_shortcut(
                #skipConn, l.shape[-1], 1, True
                features, features, 1, True  # first features is probably wrong
            )
            layers.append(changed_shortcut)

            #l = l + changed_shortcut
            torch_layers.append((torch.nn.Sequential(*layers), "add"))
            layers.clear()

    # end of each group need an extra activation
    if activation_function == "bnrelu":
        layers.append(features)
    if activation_function == "inrelu":
        layers.append(features)

    torch_layers.append((torch.nn.Sequential(*layers), "end"))

    return torch_layers

    #if getLongSkipFrom is not None:
    #    return l, skipConnection
    #else:
    #    return l