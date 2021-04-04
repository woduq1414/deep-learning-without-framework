# %%

import sys

import framework.layer as Layer
import framework.optimizer as Optimizer
from framework.functions import *
from framework.network import MultiLayerNet


from dataset.mnist import load_mnist

import numpy as np


def make_sample_data_set():
    """
        x는 0~999999 정수이다.
        정답은 x를 100000으로 나눴을 때의 몫이다.
    """

    x = np.random.randint(999999, size=(10000, 1))

    t_data = x.flatten() // 100000

    # t_data 원핫 인코딩 코드
    num = np.unique(t_data, axis=0)
    num = num.shape[0]
    t = np.eye(num)[t_data]

    return x, t


def make_sample_data_set_regression():
    x = np.random.randint(9, size=(300, 1))
    y = np.dot(x, np.array([3]))
    t = np.reshape(y, (y.shape[0], 1))

    return x, t


def make_sample_data_set_regression2():
    x = np.random.randint(999999, size=(300, 2))
    y = np.dot(x, np.array([3, -2])) + 7
    #     y =  y + (4 * np.random.random_sample((1,y.shape[0])) - 2).flatten()
    t = np.reshape(y, (y.shape[0], 1))

    return x, t


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)

x_data = np.append(x_train, x_test, axis=0)
t_data = np.append(t_train, t_test, axis=0)

net = MultiLayerNet(is_use_dropout=True, dropout_ratio=0.2)
net.add_layer(Layer.Dense(64, input_size=784))
net.add_layer(Layer.BatchNormalization())
net.add_layer(Layer.Relu())
net.add_layer(Layer.Dense(64))
net.add_layer(Layer.BatchNormalization())
net.add_layer(Layer.Relu())
net.add_layer(Layer.Dense(64))
net.add_layer(Layer.BatchNormalization())
net.add_layer(Layer.Relu())
net.add_layer(Layer.Dense(64))
net.add_layer(Layer.BatchNormalization())
net.add_layer(Layer.Relu())
net.add_layer(Layer.Dense(64))
net.add_layer(Layer.BatchNormalization())
net.add_layer(Layer.Relu())
net.add_layer(Layer.Dense(10, activation=Layer.SoftmaxWithLoss()))

result = net.train(
    x_train, t_train, x_test, t_test, batch_size=300, iters_num=4000, print_epoch=1,
    optimizer=Optimizer.Adam(lr=0.01))

print("done!")


