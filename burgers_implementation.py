# -*- coding: utf-8 -*-
# @Time    : 2020/12/20 22:16
# @Author  : zhaoxiaoyu
# @File    : burgers.py

from torch import nn, optim
import torch
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pi_net import PINNNet

# construct neural network to approximate u(t, x)
class PhysicsInfromedNN(nn.Module):
    def __init__(self, layers):
        super(PhysicsInfromedNN, self).__init__()
        self.net_u = self.neural_net(layers)

    def forward(self, inputs):
        u = self.net_u(inputs)
        return u

    def neural_net(self, layers):
        num_layers = len(layers)
        layer_list = []
        for i in range(num_layers - 2):
            layer_list += [
                nn.Linear(layers[i], layers[i + 1]),
                nn.Tanh()
            ]
        layer_list += [
            nn.Linear(layers[-2], layers[-1]),
        ]
        return nn.Sequential(*layer_list)


# compute the gradient of x and t, satisfy the Burger's equation
def net_f(net, inputs, nu):
    x, t = inputs[:, 0:1].requires_grad_(), inputs[:, 1:].requires_grad_()
    u = net(torch.cat((x, t), dim=1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    return f


def prepare_data(data_path, N_u, N_f):
    """
    :param data_path:   .mat文件路径
    :param N_u:         N_u采样点数量
    :param N_f:         N_f采样点数量
    :return:
    X_u_train:  边界和初始条件区域采样点
    u_train:    边界和初始条件区域采样点的值
    X_f_train:  求解区域配置点
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


def train(net, net_f, optimizer, criterion, train_data, epoch_cur, nu):
    X_u_train, u_train, X_f_train = train_data
    X_u_train, u_train, X_f_train = X_u_train.cuda(), u_train.cuda(), X_f_train.cuda()

    def closure():
        optimizer.zero_grad()

        u_pre = net(X_u_train)
        f_pre = net_f(net, X_f_train, nu)

        loss = criterion(u_pre, u_train) + criterion(f_pre, torch.zeros_like(f_pre))
        loss.backward()
        print(epoch_cur, ':', loss.item())
        return loss
    optimizer.step(closure)


def prediction(net, net_f, pre_data, nu):
    X_star, u_star, Exact, X, T, x, t, X_u_train, u_train = pre_data

    X_star_cuda = torch.Tensor(X_star).cuda()

    u_pred = net(X_star_cuda)
    f_pred = net_f(net, X_star_cuda, nu)

    u_pred, f_pred = u_pred.cpu().detach().numpy(), f_pred.cpu().detach().numpy()

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='data (%d points)' % (u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$u(t,x)$', fontsize=10)

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=10)

    savefig('figures/Burgers')


def main():
    nu = 0.01 / np.pi
    noise = 0.0

    N_u, N_f = 100, 10000
    data_path = 'Data/burgers_shock.mat'
    train_data, pre_data = prepare_data(data_path, N_u, N_f)

    # net = PhysicsInfromedNN(layers=[2, 20, 20, 20, 20, 1]).cuda()
    net = PINNNet(g_layers=[2, 20, 20, 20, 20, 1], norm='batchnorm', concat_injection=True).cuda()
    optimizer = optim.LBFGS(net.parameters(), lr=0.1, max_iter=20, tolerance_grad=1e-10)
    # optimizer = optim.Adam(net.parameters(), lr=0.1)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800], gamma=0.1)
    criterion = nn.MSELoss()

    epoch_num = 200
    start_time = time.time()
    for epoch in range(epoch_num):
        train(net, net_f, optimizer, criterion, train_data, epoch, nu)
        # scheduler.step()
    torch.save(net, 'model.pth')

    elapsed = time.time() - start_time
    print("Training time: %.4f" % elapsed)

    prediction(net, net_f, pre_data, nu)


if __name__ == '__main__':
    main()
