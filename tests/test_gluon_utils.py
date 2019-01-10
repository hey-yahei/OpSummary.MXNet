#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet.gluon.model_zoo.vision import alexnet
from mxnet.gluon.model_zoo.vision import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from mxnet.gluon.model_zoo.vision import inception_v3
from mxnet.gluon.model_zoo.vision import resnet18_v1, resnet34_v1, resnet50_v1, resnet101_v1, resnet152_v1, \
                                          resnet18_v2, resnet34_v2, resnet50_v2, resnet101_v2, resnet152_v2
from mxnet.gluon.model_zoo.vision import densenet121, densenet161, densenet169, densenet201
from mxnet.gluon.model_zoo.vision import mobilenet1_0, mobilenet0_75, mobilenet0_5, mobilenet0_25, \
                                          mobilenet_v2_1_0, mobilenet_v2_0_75, mobilenet_v2_0_5, mobilenet_v2_0_25
from mxnet.gluon.model_zoo.vision import squeezenet1_0, squeezenet1_1
from mxop.gluon import op_summary

dropped_layers = {
    alexnet: {"features": (8,), "output": None},    # Flatten - Dense - Dropout - Dense - Dropout - Dense
    vgg11: {"features": (21,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg13: {"features": (25,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg16: {"features": (31,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg19: {"features": (37,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg11_bn: {"features": (29,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg13_bn: {"features": (35,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg16_bn: {"features": (44,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    vgg19_bn: {"features": (53,), "output": None},     # Dense - Dropout - Dense - Dropout - Dense
    inception_v3: {"features": (18,), "output": None},     # AvgPool - Dropout - Dense
    resnet18_v1: {"features": (8,), "output": None},     # GlobalAvgPool - Dense
    resnet34_v1: {"features": (8,), "output": None},     # GlobalAvgPool - Dense
    resnet50_v1: {"features": (8,), "output": None},     # GlobalAvgPool - Dense
    resnet101_v1: {"features": (8,), "output": None},     # GlobalAvgPool - Dense
    resnet152_v1: {"features": (8,), "output": None},     # GlobalAvgPool - Dense
    resnet18_v2: {"features": (11,), "output": None},     # GlobalAvgPool - Flatten - Dense
    resnet34_v2: {"features": (11,), "output": None},     # GlobalAvgPool - Flatten - Dense
    resnet50_v2: {"features": (11,), "output": None},  # GlobalAvgPool - Flatten - Dense
    resnet101_v2: {"features": (11,), "output": None},  # GlobalAvgPool - Flatten - Dense
    resnet152_v2: {"features": (11,), "output": None},  # GlobalAvgPool - Flatten - Dense
    densenet121: {"features": (13,), "output": None},     # AvgPool2D - Flatten - Dense
    densenet161: {"features": (13,), "output": None},     # AvgPool2D - Flatten - Dense
    densenet169: {"features": (13,), "output": None},     # AvgPool2D - Flatten - Dense
    densenet201: {"features": (13,), "output": None},     # AvgPool2D - Flatten - Dense
    mobilenet1_0: {"features": (81,), "output": None},     # GlobalAvgPool - Flatten - Dense
    mobilenet0_75: {"features": (81,), "output": None},     # GlobalAvgPool - Flatten - Dense
    mobilenet0_5: {"features": (81,), "output": None},     # GlobalAvgPool - Flatten - Dense
    mobilenet0_25: {"features": (81,), "output": None},     # GlobalAvgPool - Flatten - Dense
    mobilenet_v2_1_0: {"features": 23, "output": (0,)},     # GlobalAvgPool - Conv2D - Flatten
    mobilenet_v2_0_75: {"features": 23, "output": (0,)},     # GlobalAvgPool - Conv2D - Flatten
    mobilenet_v2_0_5: {"features": 23, "output": (0,)},     # GlobalAvgPool - Conv2D - Flatten
    mobilenet_v2_0_25: {"features": 23, "output": (0,)},     # GlobalAvgPool - Conv2D - Flatten
    squeezenet1_0: {"output": (0,)},     # Conv2D - Activation - AvgPool2D - Flatten
    squeezenet1_1: {"output": (0,)},     # Conv2D - Activation - AvgPool2D - Flatten
}


def _fetch_dropped_fc(func, net):
    ret = []
    dropped = dropped_layers.get(func, [])
    for k, v in dropped.items():
        if type(v) is tuple:
            if len(v) == 1:
                ret.extend(getattr(net, k)[v[0]:])
            elif len(v) == 2:
                ret.extend(getattr(net, k)[v[0]:v[1]])
        elif type(v) is int:
            ret.append(getattr(net, k)[v])
        elif v is None:
            ret.append(getattr(net, k))
    return ret


def test_op_summary(m, input_size=(1,3,224,224), drop_fc=False):
    print("test for", m.__name__)
    net = m()
    net.initialize()
    op_summary(net, input_size, exclude=_fetch_dropped_fc(m, net) if drop_fc else [])
    print()


if __name__ == "__main__":
    drop_fc = True

    test_op_summary(alexnet, drop_fc=drop_fc)

    test_op_summary(vgg11, drop_fc=drop_fc)
    test_op_summary(vgg13, drop_fc=drop_fc)
    test_op_summary(vgg16, drop_fc=drop_fc)
    test_op_summary(vgg19, drop_fc=drop_fc)

    test_op_summary(vgg11_bn, drop_fc=drop_fc)
    test_op_summary(vgg13_bn, drop_fc=drop_fc)
    test_op_summary(vgg16_bn, drop_fc=drop_fc)
    test_op_summary(vgg19_bn, drop_fc=drop_fc)

    test_op_summary(inception_v3, (1,3,299,299), drop_fc=drop_fc)

    test_op_summary(resnet18_v1, drop_fc=drop_fc)
    test_op_summary(resnet34_v1, drop_fc=drop_fc)
    test_op_summary(resnet50_v1, drop_fc=drop_fc)
    test_op_summary(resnet101_v1, drop_fc=drop_fc)
    test_op_summary(resnet152_v1, drop_fc=drop_fc)

    test_op_summary(resnet18_v2, drop_fc=drop_fc)
    test_op_summary(resnet34_v2, drop_fc=drop_fc)
    test_op_summary(resnet50_v2, drop_fc=drop_fc)
    test_op_summary(resnet101_v2, drop_fc=drop_fc)
    test_op_summary(resnet152_v2, drop_fc=drop_fc)

    test_op_summary(densenet121, drop_fc=drop_fc)
    test_op_summary(densenet161, drop_fc=drop_fc)
    test_op_summary(densenet169, drop_fc=drop_fc)
    test_op_summary(densenet201, drop_fc=drop_fc)

    test_op_summary(mobilenet1_0, drop_fc=drop_fc)
    test_op_summary(mobilenet0_75, drop_fc=drop_fc)
    test_op_summary(mobilenet0_5, drop_fc=drop_fc)
    test_op_summary(mobilenet0_25, drop_fc=drop_fc)

    test_op_summary(mobilenet_v2_1_0, drop_fc=drop_fc)
    test_op_summary(mobilenet_v2_0_75, drop_fc=drop_fc)
    test_op_summary(mobilenet_v2_0_5, drop_fc=drop_fc)
    test_op_summary(mobilenet_v2_0_25, drop_fc=drop_fc)

    test_op_summary(squeezenet1_0, drop_fc=drop_fc)
    test_op_summary(squeezenet1_1, drop_fc=drop_fc)
