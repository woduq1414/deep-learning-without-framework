# %%

import sys

import framework.layer as Layer
import framework.optimizer as Optimizer
import framework.initializer as Initializer
from framework.functions import *
from framework.network import MultiLayerNet
import framework.scaler as Scaler
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
    x = np.random.randint(3 , size=(300, 1))
    y = np.dot(x, np.array([2]))
    t = np.reshape(y, (y.shape[0], 1))

    return x, t


def make_sample_data_set_regression2():
    x = np.random.randint(999999, size=(300, 2))
    y = np.dot(x, np.array([3, -2])) + 7
    #     y =  y + (4 * np.random.random_sample((1,y.shape[0])) - 2).flatten()
    t = np.reshape(y, (y.shape[0], 1))

    return x, t

def make_sample_data_set_regression3():
    x = np.random.rand(300, 2) * 3
    y = np.dot(x, np.array([2, 1])) + 3
    #     y =  y + (4 * np.random.random_sample((1,y.shape[0])) - 2).flatten()
    t = np.reshape(y, (y.shape[0], 1))

    # print(x)

    return x, t



x_data, t_data = make_sample_data_set_regression3()

print(x_data[:3])
print(t_data[:3])

net = MultiLayerNet()
net.add_layer(Layer.Dense(1, input_size = 2, activation=Layer.IdentityWithLoss() ))
# net.add_layer(Layer.Dense(5, input_size = 2, activation=Layer.Relu() ))
# net.add_layer(Layer.Dense(1))

x_train, t_train, x_test, t_test = shuffle_split_data(x_data, t_data, 0.2)

print(net.params)

# scaler = Scaler.StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

result = net.train(
        x_train, t_train, x_test, t_test, batch_size = 100, iters_num = 1000, print_epoch = 30,
        optimizer = Optimizer.SGD(lr=0.001)
)


print("done!")

print(net.params)
print(net.layers["Affine0"].__dict__)


net.save_model()

# net.load_model("weight.npz")