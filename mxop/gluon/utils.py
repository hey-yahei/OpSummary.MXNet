#-*- coding: utf-8 -*-

import logging
from mxnet import nd
from mxnet.gluon import nn
from .count_hooks import *

__all__ = ["op_summary"]
__author__ = "YaHei"

register_hooks = {
    nn.Conv2D: count_conv2d,
    nn.Dense: count_fc,
    nn.BatchNorm: count_bn,
    nn.MaxPool1D: count_maxpool,
    nn.MaxPool2D: count_maxpool,
    nn.MaxPool3D: count_maxpool,
    nn.AvgPool1D: count_avgpool,
    nn.AvgPool2D: count_avgpool,
    nn.AvgPool3D: count_avgpool,
    nn.Dropout: None,
    nn.Activation: None    # ignore relu and relu6
}

_ops2collect = ["adds", "muls", "divs", "exps"]
_ops_dict = tuple( zip(_ops2collect, [0]*len(_ops2collect)) )

def _accumulate_ops(m):
    if not hasattr(_accumulate_ops, 'total_ops'):
        _accumulate_ops.total_ops = dict(_ops_dict)
    if hasattr(m, "ops"):
        for op_name in _ops2collect:
            _accumulate_ops.total_ops[op_name] += m.ops[op_name]

def _clear_accumulator():
    del _accumulate_ops.total_ops

def op_summary(net, input_size, custom_ops={}):
    def add_hooks(m):
        m_type = type(m)
        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
        elif "_" in m.name:
            logging.info("No count functions match for ", m)

        if fn is not None:
            m.ops = dict(_ops_dict)
            m.register_forward_hook(fn)

    net.apply(add_hooks)
    __ = net(nd.zeros(shape=input_size))
    net.apply(_accumulate_ops)

    # print(_accumulate_ops.total_ops)
    for op_type, num in _accumulate_ops.total_ops.items():
        print("{}: {:,}".format(op_type, num))
    _clear_accumulator()
    params_counter = 0
    params = net.collect_params()
    for p in params:
        params_counter += params[p].data().size
    print("total parameters: {:,}".format(params_counter))
