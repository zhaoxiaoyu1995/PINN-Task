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


class PhysicsInfromedNN(nn.Module):
    def __init__(self, layers, lb, ub):
        super(PhysicsInfromedNN, self).__init__()
        self.net_u = self.neural_net(layers)
        self.lb = torch.Tensor(lb).cuda()
        self.ub = torch.Tensor(ub).cuda()

    def forward(self, x, t):
        x.requires_grad_()
        t.requires_grad_()
        X = torch.cat((x, t), dim=1)
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        uv = self.net_u(X)
        u = uv[:, 0:1]
        v = uv[:, 1:2]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        return u, v, u_x, v_x

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


def net_f_uv(net, x, t):
    u, v, u_x, v_x = net(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
    f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

    return f_u, f_v


def prepare_data(data_path, N0, N_b, N_f):
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    data = sio.loadmat(data_path)

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    ###########################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)
    return (x0, u0, v0, tb, X_f, lb, ub), \
           (Exact_h, x, t, x0, lb, ub, tb, X, T, X_star, u_star, v_star, h_star), \
           (lb, ub)


def train(net, net_f_uv, optimizer, criterion, train_data, epoch_cur):
    x0, u0, v0, tb, X_f, lb, ub = train_data

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

    x0 = torch.Tensor(X0[:, 0:1]).cuda()
    t0 = torch.Tensor(X0[:, 1:2]).cuda()

    x_lb = torch.Tensor(X_lb[:, 0:1]).cuda()
    t_lb = torch.Tensor(X_lb[:, 1:2]).cuda()

    x_ub = torch.Tensor(X_ub[:, 0:1]).cuda()
    t_ub = torch.Tensor(X_ub[:, 1:2]).cuda()

    x_f = torch.Tensor(X_f[:, 0:1]).cuda()
    t_f = torch.Tensor(X_f[:, 1:2]).cuda()

    u0 = torch.Tensor(u0).cuda()
    v0 = torch.Tensor(v0).cuda()

    def closure():
        optimizer.zero_grad()

        # tf Graphs
        u0_pred, v0_pred, _, _ = net(x0, t0)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = net(x_lb, t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = net(x_ub, t_ub)
        f_u_pred, f_v_pred = net_f_uv(net, x_f, t_f)

        # Loss
        loss = criterion(u0, u0_pred) + criterion(v0, v0_pred) + criterion(u_lb_pred, u_ub_pred) \
               + criterion(v_lb_pred, v_ub_pred) + criterion(u_x_lb_pred, u_x_ub_pred) \
               + criterion(v_x_lb_pred, v_x_ub_pred) + criterion(f_u_pred, torch.zeros_like(f_u_pred)) \
               + criterion(f_v_pred, torch.zeros_like(f_v_pred))
        loss.backward()
        print(epoch_cur, ':', loss.item())
        return loss

    optimizer.step(closure)


def prediction(net, pre_data):
    Exact_h, x, t, x0, lb, ub, tb, X, T, X_star, u_star, v_star, h_star = pre_data

    X_star_cuda = torch.Tensor(X_star).cuda()
    x_input, t_input = X_star_cuda[:, 0:1], X_star_cuda[:, 1:2]
    u_pred, v_pred, _, _ = net(x_input, t_input)
    x_input, t_input = X_star_cuda[:, 0:1], X_star_cuda[:, 1:2]
    f_u_pred, f_v_pred = net_f_uv(net, x_input, t_input)

    u_pred, v_pred = u_pred.cpu().detach().numpy(), v_pred.cpu().detach().numpy()
    f_u_pred, f_v_pred = f_u_pred.cpu().detach().numpy(), f_v_pred.cpu().detach().numpy()
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, 'k--', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[:, 75], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[75]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact_h[:, 100], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact_h[:, 125], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[125, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[125]), fontsize=10)

    savefig('figures/NLS')


def main():
    N0 = 50
    N_b = 50
    N_f = 20000
    data_path = 'data/NLS.mat'
    train_data, pre_data, lb_ub = prepare_data(data_path, N0, N_b, N_f)

    net = PhysicsInfromedNN(layers=[2, 100, 100, 100, 100, 2], lb=lb_ub[0], ub=lb_ub[1]).cuda()
    optimizer = optim.LBFGS(net.parameters(), lr=0.1, max_iter=20, tolerance_grad=1e-08)
    criterion = nn.MSELoss()

    epoch_num = 200
    start_time = time.time()
    for epoch in range(epoch_num):
        train(net, net_f_uv, optimizer, criterion, train_data, epoch)
    torch.save(net, 'model.pth')

    elapsed = time.time() - start_time
    print("Training time: %.4f" % elapsed)

    prediction(net, pre_data)


if __name__ == '__main__':
    main()
