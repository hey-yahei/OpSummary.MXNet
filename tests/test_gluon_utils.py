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

def test_op_summary(m, input_size=(1,3,224,224)):
    print("test for", m.__name__)
    net = m()
    net.initialize()
    op_summary(net, input_size)
    print()

if __name__ == "__main__":
    test_op_summary(alexnet)

    test_op_summary(vgg11)
    test_op_summary(vgg13)
    test_op_summary(vgg16)
    test_op_summary(vgg19)

    test_op_summary(vgg11_bn)
    test_op_summary(vgg13_bn)
    test_op_summary(vgg16_bn)
    test_op_summary(vgg19_bn)

    test_op_summary(inception_v3, (1,3,299,299))

    test_op_summary(resnet18_v1)
    test_op_summary(resnet34_v1)
    test_op_summary(resnet50_v1)
    test_op_summary(resnet101_v1)
    test_op_summary(resnet152_v1)

    test_op_summary(resnet18_v2)
    test_op_summary(resnet34_v2)
    test_op_summary(resnet50_v2)
    test_op_summary(resnet101_v2)
    test_op_summary(resnet152_v2)

    test_op_summary(densenet121)
    test_op_summary(densenet161)
    test_op_summary(densenet169)
    test_op_summary(densenet201)

    test_op_summary(mobilenet1_0)
    test_op_summary(mobilenet0_75)
    test_op_summary(mobilenet0_5)
    test_op_summary(mobilenet0_25)

    test_op_summary(mobilenet_v2_1_0)
    test_op_summary(mobilenet_v2_0_75)
    test_op_summary(mobilenet_v2_0_5)
    test_op_summary(mobilenet_v2_0_25)

    test_op_summary(squeezenet1_0)
    test_op_summary(squeezenet1_1)
