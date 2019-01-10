#-*- coding: utf-8 -*-

import sys
sys.path.append("..")

from mxnet import nd
from mxnet.gluon.nn import Conv2D, BatchNorm, AvgPool2D, AvgPool3D, Dense

from mxop.gluon.count_hooks import *

_ops2collect = ["adds", "muls", "divs", "exps"]
_ops_dict = tuple( zip(_ops2collect, [0]*len(_ops2collect)) )


def test_counter(inputs, m, hook):
    print("test function: ", hook.__name__)
    m.ops = dict(_ops_dict)
    m.register_forward_hook(hook)
    print("input:", inputs.shape)
    print("output:", m(inputs).shape )
    for op_type, num in m.ops.items():
        print("{}: {:,}".format(op_type, num))
    print()


if __name__ == "__main__":
    inputs = nd.zeros(shape=(2,3,10,20))

    m = Conv2D(channels=10, in_channels=3, strides=1, padding=1, use_bias=True, kernel_size=5)
    m.initialize()
    test_counter(inputs, m, count_conv2d)

    m = BatchNorm()
    m.initialize()
    test_counter(inputs, m, count_bn)

    m = AvgPool2D()
    m.initialize()
    test_counter(inputs, m, count_avgpool)

    m = AvgPool3D()
    m.initialize()
    test_counter(nd.zeros(shape=(2,3,10,10,20)), m, count_avgpool)

    m = Dense(in_units=10, units=20)
    m.initialize()
    test_counter(nd.zeros(shape=(2,10)), m, count_fc)