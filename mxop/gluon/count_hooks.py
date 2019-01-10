#-*- coding: utf-8 -*-

__all__ = [
    "count_conv2d", "count_avgpool", "count_bn", "count_fc", "count_maxpool",
    # "count_relu", "count_softmax",
]
__author__ = "YaHei"

import functools


def count_conv2d(m, x, y):
    x = x[0]

    cout, cin_g, kw, kh = m.weight.shape
    batch_size, cin, inw, inh = x.shape
    __, __, outw, outh = y.shape
    groups = cin // cin_g

    muls_per_elem = kw * kh * cin // groups
    adds_per_elem = kw * kh * cin // groups - 1
    if m.bias is not None:
        adds_per_elem += 1

    n_elem = batch_size * outw * outh * cout
    m.ops["muls"] = n_elem * muls_per_elem
    m.ops["adds"] = n_elem * adds_per_elem
    m.ops["divs"] = 0
    m.ops["exps"] = 0


def count_fc(m, x, y):
    x = x[0]

    n_features = functools.reduce(lambda x,y: x*y, x.shape[1:], 1) - 1
    adds_per_elem = n_features - 1
    muls_per_elem = n_features

    n_elem = y.size
    m.ops["muls"] = muls_per_elem * n_elem
    m.ops["adds"] = adds_per_elem * n_elem
    m.ops["divs"] = 0
    m.ops["exps"] = 0


def count_bn(m, x, y):
    x = x[0]

    # out_bn = ( gamma * (out-mean) ) / sqrt(variance) + beta
    n_elem = x.size
    m.ops["adds"] = 2*n_elem
    m.ops["muls"] = n_elem
    m.ops["divs"] = n_elem   # m.ops["divs"] = 0 if merge $gamma$ and $sqrt(variance)$
    m.ops["exps"] = 0


# def count_relu(m, x, y):
#     # ignore
#     # out_relu = max(0, out)
#     m.ops["adds"] = 0
#     m.ops["muls"] = 0
#     m.ops["divs"] = 0
#     m.ops["exps"] = 0


# def count_softmax(m, x, y):
#     x = x[0]
#
#     # deno = sum(e^{x})
#     # y = e^{x} / deno
#     batch_size, n_features = x.shape
#     adds_per_batch = n_features - 1
#     divs_per_batch = n_features
#     exps_per_batch = n_features
#
#     m.ops["adds"] = adds_per_batch * batch_size
#     m.ops["muls"] = 0
#     m.ops["divs"] = divs_per_batch * batch_size
#     m.ops["exps"] = exps_per_batch * batch_size


def count_maxpool(m, x, y):
    # ignore
    # y = max(0, x[a:b,c:d])
    m.ops["adds"] = 0
    m.ops["muls"] = 0
    m.ops["divs"] = 0
    m.ops["exps"] = 0


def count_avgpool(m, x, y):
    x = x[0]

    m_str = str(m)  # AvgPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
    m_str = m_str[m_str.index("size"):]     # size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
    m_str = m_str[len("size=(") : m_str.index(")")]    # 2, 2
    kernel = [int(k) for k in m_str.split(",")]
    adds_per_elem = functools.reduce(lambda x,y: x*y, kernel, 1) - 1
    divs_per_elem = 1

    n_elem = y.size
    m.ops["adds"] = adds_per_elem * n_elem
    m.ops["muls"] = 0
    m.ops["divs"] = divs_per_elem * n_elem
    m.ops["exps"] = 0
