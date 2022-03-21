# PINN-Task
近些年，深度学习在多个学科上取得了巨大的成功，标志性的学科包括：图像识别、自然语言处理、语音识别、生物医学等。但是对于一些复杂的物理、生物和工程系统，获取合适规模的标注数据通常是困难的，先进的机器学习技术性能面临着巨大的挑战。

例如，采用深度学习方法，利用少量的标注数据学习高维的输入和输出数据之间的非线性映射似乎是难以完成的任务。但是对于一个物理系统，存在着大量的先验知识，最直观的方式是利用这些知识辅助深度神经网络的训练。这些先验知识可以是控制系统的动力学物理定律、经验验证的规则或其他领域的专业知识，利用先验信息作为正则化，限制模型搜索的解空间范围。

**PINN（Physics-Informed Neural Network）**将物理知识（比如物理定律，PDE，或者简化的数学模型）嵌入神经网络，从而设计出更好的机器学习模型。这些模型可以自动满足一些物理不变量，可以被更快地训练，达到更好的准确性。

# 1. 偏微分方程

科学计算在各门自然科学（物理学、气象学、地质学和生命科学等）和技术科学与工程科学（核技术、石油勘探、航空与航天和大型土木工程等）中起着越来越大的作用，在很多重要领域中成为不可缺少的工具。而科学与工程计算中最重要的内容就是求解在科学研究和工程技术中出现的各种各样的偏微分方程或方程组。

含有未知函数![img](https://cdn.nlark.com/yuque/__latex/d096fb72a1e891c4c90a62a3b548e51e.svg)的偏导数的方程称为偏微分方程。这里![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)是![img](https://cdn.nlark.com/yuque/__latex/40b85027598d87611b1c8d5d11e46812.svg)个自变量的函数。在很多应用问题中，专门用![img](https://cdn.nlark.com/yuque/__latex/e358efa489f58062f10dd7316b65649e.svg)表示时间变量，![img](https://cdn.nlark.com/yuque/__latex/50add2d36dfb3c4c74d937f874b81295.svg)表示空间变量。

常见的偏微分方程例子：

（1）Poisson方程

泊松方程是数学中一个常见于[静电学](https://baike.baidu.com/item/静电学/10485691)、机械工程和[理论物理](https://baike.baidu.com/item/理论物理/2490260)的[偏微分方程](https://baike.baidu.com/item/偏微分方程/818038)。形式如下：

![img](https://cdn.nlark.com/yuque/__latex/b8fe23a826da0067a86a4174c8bee88f.svg)

（2）热传导方程

热传导方程（或称热方程）是一个重要的[偏微分方程](https://baike.baidu.com/item/偏微分方程/818038)，它描述一个区域内的温度如何随时间变化。

![img](https://cdn.nlark.com/yuque/__latex/c4e534e163fa23d82a714853d6b013c1.svg)

（3）波动方程

波动方程主要描述自然界中的各种的[波动](https://baike.baidu.com/item/波动/4741381)现象，包括横波和纵波，例如[声波](https://baike.baidu.com/item/声波/35769)、[光波](https://baike.baidu.com/item/光波/10730221)和[水波](https://baike.baidu.com/item/水波/1742563)。波动方程抽象自[声学](https://baike.baidu.com/item/声学/473275)，[电磁学](https://baike.baidu.com/item/电磁学/381578)，和[流体力学](https://baike.baidu.com/item/流体力学/620604)等领域。一般形式如下：

![img](https://cdn.nlark.com/yuque/__latex/5e82aaf998d7e693b58c097ff4191c1a.svg)

# 2. 神经网络自动求导技术

[神经网络](https://so.csdn.net/so/search?q=神经网络&spm=1001.2101.3001.7020)可以看作一个复合数学函数，网络结构设计决定了多个基础函数如何复合成复合函数，网络的训练过程确定了复合函数的所有参数。为了获得一个“优秀”的函数，训练过程中会基于给定的数据集合，对该函数参数进行多次迭代修正。

神经网络上的自动求导基于计算图构建和链式求导实现。深度学习框架如pytorch包含自动求导的包，可以直接调用。PyTorch中所有神经网络的核心是autograd包。autograd包为张量上的所有操作提供自动微分。

例如：

```python
In [1]:		x = torch.Tensor(1)
In [2]: 	x.requires_grad=True
In [3]:		y = x * 2
In [4]:		grad = torch.autograd.grad(y, x)

Out [4]:	grad=2
```

如果将神经网络看作一个通用的函数拟合器拟合![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)，同时![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)对于各个变量![img](https://cdn.nlark.com/yuque/__latex/aafd02a1e758c1aec3177ba7115a81b4.svg)的偏导可以通过神经网络自动求导实现。

**利用偏微分方程构建损失函数指导神经网络拟合的**![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)**满足偏微分方程描述的**![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)**和偏导之间的关系，这是基于自动微分PINN中最朴素的思想。**

# 3. PINN形式化描述

**Physics-informed neural networks (PINN) 具体的数学描述如下：**

**问题：**

![img](https://cdn.nlark.com/yuque/__latex/81694351e383556a73e1732ef0fdfcea.svg)

- 初始\边界条件 ：![img](https://cdn.nlark.com/yuque/__latex/555242dfdd6cf5fb17b5f1432a41b3e2.svg)
- 其他信息：![img](https://cdn.nlark.com/yuque/__latex/1e645af190b78980d7140ae1d9a30a74.svg)

损失函数：

![img](https://cdn.nlark.com/yuque/__latex/0eef49d6962be0f38948d3cd16dbdb49.svg)

其中：

![img](https://cdn.nlark.com/yuque/__latex/f14e957f278120863fdbcc224c5aaefb.svg)

![img](https://cdn.nlark.com/yuque/__latex/b217a1470463f2266a017ddca3680e87.svg)

![img](https://cdn.nlark.com/yuque/__latex/43b992f258ebf92b805dcbd7a2e3aa69.svg)

# 4. 示例：PINN求解Schrodinger方程

一维非线性薛定谔方程是一个经典的场方程，用于研究量子力学系统，包括非线性波在光纤、导波管波导中的传播、等离子体波等。

示例为带有循环边界条件的一维薛定谔方程如下：

![img](https://cdn.nlark.com/yuque/__latex/91e6102b0224a30206e62b833c5240f9.svg)

其中微分方程的残差项为：

![img](https://cdn.nlark.com/yuque/__latex/837ad280520697ee668ee8fc57ba1c48.svg)

初始条件：

![img](https://cdn.nlark.com/yuque/__latex/45ccd767efc9c06d05a8c52101b6ee31.svg)

边界条件：

![img](https://cdn.nlark.com/yuque/__latex/a4761c5b59a72c8d97db0ea8e5ff3156.svg)

其中神经网络拟合的![img](https://cdn.nlark.com/yuque/__latex/7a5f0b0086819fb131b2e04d1aa5c11e.svg)为一个复数，包含实部和虚部![img](https://cdn.nlark.com/yuque/__latex/927a4fa3dadfc9ecc3750a72bc200fb4.svg)，可以采用包含两个输出的神经网络拟合，即全连接神经网络的输出层神经单元数量为2.

**首先，对于解**![img](https://cdn.nlark.com/yuque/__latex/2510c39011c5be704182423e3a695e91.svg)**可以采用最简单的全连接神经网络拟合，神经网络的输入为**![img](https://cdn.nlark.com/yuque/__latex/b69cf130348b5d53a7aaeb7f79eea645.svg)**，输出为对应时间和坐标处的函数值的实部**![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)**和虚部**![img](https://cdn.nlark.com/yuque/__latex/9e3669d19b675bd57058fd4664205d2a.svg)**。**

```python
class PhysicsInfromedNN(nn.Module):
    def __init__(self, layers, lb, ub):
        super(PhysicsInfromedNN, self).__init__()
        self.net_u = self.neural_net(layers)
        self.lb = torch.Tensor(lb).cuda()
        self.ub = torch.Tensor(ub).cuda()
        # self.net_v = self.neural_net(layers)

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
```

使用pytorch自动微分求解![img](https://cdn.nlark.com/yuque/__latex/703f259d2f1a609d112d61c642e0f17d.svg)对![img](https://cdn.nlark.com/yuque/__latex/e358efa489f58062f10dd7316b65649e.svg)的一阶偏导![img](https://cdn.nlark.com/yuque/__latex/1a558f966ee0d9665a5f718362c4013a.svg)，对![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)的一阶偏导![img](https://cdn.nlark.com/yuque/__latex/941e765f3286cd2b922912cd6d55572d.svg)和对![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)的二阶偏导![img](https://cdn.nlark.com/yuque/__latex/814361d90c6580b35b2d453ffc13fe79.svg)。

```python
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
```

通过偏微分方程、边界条件、初始条件中![img](https://cdn.nlark.com/yuque/__latex/fd937b1240213e39dcaf90e3b397b61b.svg)之间的关系构建损失函数进行网络训练。

任务中的Loss包含三部分：

- **初始条件loss**

从初始条件区域采点，这些区域点的值已知，使得网络预测与这点保持一致。

![img](https://cdn.nlark.com/yuque/__latex/a9495d63bdf380e42bf5662c78af75ef.svg)

其中![img](https://cdn.nlark.com/yuque/__latex/7d4bf4f712c97b47d3a78ab48574ae62.svg)为在初始条件区域![img](https://cdn.nlark.com/yuque/__latex/ba9cf3b28d44206d633d11ccd540be21.svg)中采样的离散点的数量。

```python
u0_pred, v0_pred, _, _ = net(x0, t0)
loss_0 = criterion(u0, u0_pred) + criterion(v0, v0_pred)
```

- **循环边界条件loss**

在边界条件区域，边界区域![img](https://cdn.nlark.com/yuque/__latex/839e77a79e39db2660c1d64c1161d8d0.svg)和边界区域![img](https://cdn.nlark.com/yuque/__latex/3232ffe72034023430f2b2da728be17e.svg)上需要满足值和一阶偏导值相等。

![img](https://cdn.nlark.com/yuque/__latex/2f31e22ae91b64fa9f87f2cbb3fdd6ca.svg)

其中![img](https://cdn.nlark.com/yuque/__latex/8c77c30b1b27cdcd43fa62e8764c31ab.svg)为在边界区域![img](https://cdn.nlark.com/yuque/__latex/839e77a79e39db2660c1d64c1161d8d0.svg)和边界区域![img](https://cdn.nlark.com/yuque/__latex/3232ffe72034023430f2b2da728be17e.svg)中采样的离散点的数量。

```python
u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = net(x_lb, t_lb)
u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = net(x_ub, t_ub)

loss_b = criterion(u_lb_pred, u_ub_pred) + criterion(v_lb_pred, v_ub_pred) 
		+ criterion(u_x_lb_pred, u_x_ub_pred) + criterion(v_x_lb_pred, v_x_ub_pred)
```

- **残差loss**

![img](https://cdn.nlark.com/yuque/__latex/b9d4c80e41c00d445d94ea30d1442815.svg)

其中：

![img](https://cdn.nlark.com/yuque/__latex/837ad280520697ee668ee8fc57ba1c48.svg)

其中![img](https://cdn.nlark.com/yuque/__latex/a75707024f355a0ae7b07970a62388a9.svg)为在求解区域![img](https://cdn.nlark.com/yuque/__latex/4ac69198bbf3ad670c131b6378be035d.svg)中采样的离散点的数量。

```python
def net_f_uv(net, x, t):
    u, v, u_x, v_x = net(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
    f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

    return f_u, f_v

f_u_pred, f_v_pred = net_f_uv(net, x_f, t_f)
loss_f = criterion(f_u_pred, torch.zeros_like(f_u_pred)) + criterion(f_v_pred, torch.zeros_like(f_v_pred))
```

- **有监督损失函数（如果有观测值）**

从除初始和边界条件等已知区域采点，这些区域点作为观测值，使得网络预测与已知观测值保持一致。

![img](https://cdn.nlark.com/yuque/__latex/83788df8f651873b5ea950fcf6d8061d.svg)

其中![img](https://cdn.nlark.com/yuque/__latex/ae0eef8cb3bbd510703902a0f9ea2e22.svg)为在求解区域![img](https://cdn.nlark.com/yuque/__latex/4ac69198bbf3ad670c131b6378be035d.svg)中观测值的数量。

在PINN中，大多数采用L-BFGS优化器进行训练。采用5层全连接神经网络，其中隐层神经单元数量为100。采用tanh激活函数。从初始边界采样50个点、循环边界采样50个点，求解区域中采样20000个配置点，最终训练loss为：

![img](https://cdn.nlark.com/yuque/__latex/0cb1fa0a7e125bc65943de11e9066b1a.svg)

- **训练结果展示**

最终网络训练效果如下，其中上部分为整个时空区域解![img](https://cdn.nlark.com/yuque/__latex/f369d8181d309e801fb55755090033be.svg)的可视化展示，下半部分分别为0.59、0.79和0.98时刻求解结果。

![img](https://cdn.nlark.com/yuque/0/2022/png/2753467/1647826114051-4c19905a-0cf0-4bca-8f83-7ca80a61dc1a.png)

**薛定谔方程求解示例在schrodinger.py文件中已给出。**

**详细了解PINN和Schrodinger's方程求解，请仔细阅读参考文献**[**Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations**](https://arxiv.53yu.com/pdf/1711.10561)

# 4. 任务：PINN求解Burgers方程

Burgers方程出现在应用数学的各个领域，包括流体力学、非线性声学、气体动力学和交通流等。这是一个基础的偏微分方程，可以从Navier-Stokes方程中去掉压力梯度项而得到。

本示例求解如下带有狄利克雷边界条件的一维Burgers方程。

![img](https://cdn.nlark.com/yuque/__latex/80dbed259565469883ba773242e888c3.svg)

其中微分方程的残差项为：

![img](https://cdn.nlark.com/yuque/__latex/741f5b035edc351395d5b62abe060983.svg)

初始条件：

![img](https://cdn.nlark.com/yuque/__latex/9329cc18768e51256d6838bf1c501f7b.svg)

边界条件：

![img](https://cdn.nlark.com/yuque/__latex/a2e15a19b1ab35fc1a07ba2494535f5a.svg)

**与求解薛定谔方程相比，预测值为实数而非复数，神经网络的输出维度为1，本任务更简单。**

**首先，对于解**![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)**可以采用最简单的全连接神经网络拟合，神经网络的输入为**![img](https://cdn.nlark.com/yuque/__latex/b69cf130348b5d53a7aaeb7f79eea645.svg)**，输出为对应时间和坐标处的函数值。**

**使用pytorch自动微分求解**![img](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg)**对**![img](https://cdn.nlark.com/yuque/__latex/e358efa489f58062f10dd7316b65649e.svg)**的一阶偏导**![img](https://cdn.nlark.com/yuque/__latex/92b76fbf77ebf0b86b15e0881b0a1a49.svg)**，对**![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)**的一阶偏导**![img](https://cdn.nlark.com/yuque/__latex/9a30512405c4138a689b66957d9e1601.svg)**和对**![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)**的二阶偏导**![img](https://cdn.nlark.com/yuque/__latex/d866f3038e3bc455be42f5765eb61d0c.svg)**。**

**通过偏微分方程、边界条件、初始条件中**![img](https://cdn.nlark.com/yuque/__latex/91614f244f221ae99d133154eab2cd05.svg)**之间的关系构建损失函数进行网络训练。**

- **残差项损失函数**

![img](https://cdn.nlark.com/yuque/__latex/1d9e11fa18aa55722e0b21b68440cc8b.svg)

其中![img](https://cdn.nlark.com/yuque/__latex/a75707024f355a0ae7b07970a62388a9.svg)为在求解区域![img](https://cdn.nlark.com/yuque/__latex/0afb98fcd5f83972889109b161f08a66.svg)中采样的离散点的数量。

- **初始/边界条件损失函数**

从初始和边界条件区域采点，这些区域点的值已知，使得网络预测与这点保持一致。

![img](https://cdn.nlark.com/yuque/__latex/52a02d4a53114b93748a4933a2ce60ce.svg)

其中![img](https://cdn.nlark.com/yuque/__latex/8c77c30b1b27cdcd43fa62e8764c31ab.svg)为在求解区域![img](https://cdn.nlark.com/yuque/__latex/7c2e2ad8777838a9c0d9abef66736bd6.svg)中采样的离散点的数量。



已知数据burgers_shock.mat

```plain
usol:	已知解,256x100.
t:	已知解中的时间维度值,100x1
x:	已知解中的空间坐标值,256x1
```

数据读取函数prepare_data已实现并提供使用。

**任务：**

**1.实现PINN求解Burgers方程，并可视化整个时空域上的求解结果，并展示0.30、0.60和0.90时刻求解结果。展示示例如下：**

![img](https://cdn.nlark.com/yuque/0/2022/png/2753467/1647822176532-951ccbe8-4cb1-4506-8166-21577c4d2b73.png)

**2. 讨论采样点数量对求解结果的影响，表格示例如下：**

|                                                              | ![img](https://cdn.nlark.com/yuque/__latex/a75707024f355a0ae7b07970a62388a9.svg) |      |      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ---- |
| ![img](https://cdn.nlark.com/yuque/__latex/8c77c30b1b27cdcd43fa62e8764c31ab.svg) | 3000                                                         | 6000 | 9000 |
| 40                                                           |                                                              |      |      |
| 80                                                           |                                                              |      |      |
| 160                                                          |                                                              |      |      |

**3.讨论神经网络规模对求解结果的影响，表格示例如下：**

|              | 隐层单元数 |      |      |
| ------------ | ---------- | ---- | ---- |
| 神经网络层数 | 10         | 20   | 40   |
| 2            |            |      |      |
| 4            |            |      |      |
| 8            |            |      |      |

**4.讨论优化器对求解结果的影响，表格示例如下：**

| 优化器 |      |
| ------ | ---- |
| SGD    |      |
| Adam   |      |
| L-BFGS |      |
| ....   |      |

# 参考资料

[1] [内嵌物理知识神经网络（PINN）是个坑吗？](https://zhuanlan.zhihu.com/p/468748367)

[2] [内嵌物理的深度学习](https://app6ca5octe2206.pc.xiaoe-tech.com/detail/v_61149143e4b054ed7c4d0b26/3?fromH5=true)

[3] [偏微分方程在物理学中的完美应用——热方程，推导和示例](https://baijiahao.baidu.com/s?id=1693765641862149931&wfr=spider&for=pc)

[4] [机器学习之自动求导](https://zhuanlan.zhihu.com/p/30022887)

[5] [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.53yu.com/pdf/1711.10561)

[6] [Tensorflow实现代码参考](https://github.com/maziarraissi/PINNs)

感谢以下仓库的工作。

- [PINNs](https://github.com/maziarraissi/PINNs)
