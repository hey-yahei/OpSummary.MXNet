#-*- coding: utf-8 -*-

import logging
from mxnet import nd
from mxnet.gluon import nn
from .count_hooks import *

__all__ = ["count_ops", "count_params", "op_summary"]
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
    """Accumulate op counters to static attr of this function
    This function is used at mxnet.gluon.Block.apply.
    For example, `net.apply(_accumulate_ops)`
    :param m: mxnet.gluon.Block
    :return: None
    """
    if not hasattr(_accumulate_ops, 'total_ops'):
        _accumulate_ops.total_ops = dict(_ops_dict)
    if hasattr(m, "ops"):
        for op_name in _ops2collect:
            _accumulate_ops.total_ops[op_name] += m.ops[op_name]


def _pop_accumulator(func, attr):
    """Return and clear the accumulator of function `_accumultae_ops`"""
    ret = getattr(func, attr)
    delattr(func, attr)
    return ret


def count_params(net, exclude=[]):
    """
    Count parameters for net.
    :param net: mxnet.gluon.Block
        Net or block to be counted parameters for.
    :param exclude: list of mxnet.gluon.nn.Block
        Blocks to be excluded.
    :return: int
        The number of parameters of net.
    """
    exclude_params = []
    for exc in exclude:
        exclude_params.extend(list(exc.collect_params()))

    params_counter = 0
    params = net.collect_params()
    for p in params:
        if p not in exclude_params:
            params_counter += params[p].data().size
    return params_counter


def count_ops(net, input_size, custom_ops={}, exclude=[]):
    """
    Count OPs for net.
    :param net: mxnet.gluon.Block
        Net or block to be counted parameters for.
    :param input_size: tuple
        The shape of input.
    :param custom_ops: dict with `op` as key and `func` as value, where
        `op`: class(mxnet.gluon.nn.Block), the block class you want to count.
        `func`: callable, the hook function of form `hook(block, input, output) -> None`
            In the function, you should set values for `block.op` counter.
            Ref: https://github.com/hey-yahei/OpSummary.MXNet/blob/master/mxop/gluon/count_hooks.py
    :param exclude: list of mxnet.gluon.nn.Block
        Blocks to be excluded.
    :return: dict with op_name as key and number as value
    """
    hooks = []

    # Count ops
    def _add_hooks(m):
        if m not in exclude:
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
                hooks.append( m.register_forward_hook(fn) )
    net.apply(_add_hooks)
    __ = net(nd.zeros(shape=input_size))
    net.apply(_accumulate_ops)
    op_counters = _pop_accumulator(_accumulate_ops, "total_ops")

    # Delete accumulators and detach all hooks
    def _del_ops(m):
        if hasattr(m, "ops"):
            delattr(m, "ops")
    net.apply(_del_ops)
    for h in hooks:
        h.detach()

    return op_counters


def op_summary(net, input_size, custom_ops={}, exclude=[]):
    """
    Print summary via function `count_ops` and `count_params`
    :param net: mxnet.gluon.Block
        Net or block to be counted OPs and parameters for.
    :param input_size: tuple
        The shape of input.
    :param custom_ops: dict with `op` as key and `func` as value, where
        `op`: class(mxnet.gluon.nn.Block), the block class you want to count.
        `func`: callable, the hook function of form `hook(block, input, output) -> None`
            In the function, you should set values for `block.op` counter.
            Ref: https://github.com/hey-yahei/OpSummary.MXNet/blob/master/mxop/gluon/count_hooks.py
    :param exclude: list of mxnet.gluon.nn.Block
        Blocks to be excluded.
    :return: None
    """
    op_counter = count_ops(net, input_size, custom_ops, exclude)
    for op_type, num in op_counter.items():
        print("{}: {:,}".format(op_type, num))

    param_counter = count_params(net, exclude)
    print("total parameters: {:,}".format(param_counter))
