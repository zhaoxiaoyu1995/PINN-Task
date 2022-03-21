# -*- coding: utf-8 -*-
# @Time    : 2020/12/20 22:16
# @Author  : zhaoxiaoyu
# @File    : burgers.py

import torch
import numpy as np
import scipy.io as sio
from pyDOE import lhs


def prepare_data(data_path, N_u, N_f):
    """
    :param data_path:   .mat文件路径
    :param N_u:         N_u采样点数量
    :param N_f:         N_f采样点数量
    :return:
    X_u_train:  边界和初始条件区域采样点
    u_train:    边界和初始条件区域采样点的值
    X_f_train:  求解区域配置点
    其他输出值含义请自行理解
    """
    data = sio.loadmat(data_path)

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]
    return (torch.Tensor(X_u_train), torch.Tensor(u_train), torch.Tensor(X_f_train)), \
           (X_star, u_star, Exact, X, T, x, t, X_u_train, u_train)
