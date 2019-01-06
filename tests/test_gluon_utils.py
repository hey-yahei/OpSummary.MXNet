#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet.gluon.model_zoo.vision import vgg16, vgg16_bn, resnet50_v1, mobilenet1_0
from mxop.gluon import op_summary

def test_op_summary(m, input_size=(1,3,224,224)):
    print("test for ", m.__name__)
    net = m()
    net.initialize()
    op_summary(net, input_size)
    print()

if __name__ == "__main__":
    test_op_summary(vgg16)
    test_op_summary(vgg16_bn)
    test_op_summary(resnet50_v1)
    test_op_summary(mobilenet1_0)
