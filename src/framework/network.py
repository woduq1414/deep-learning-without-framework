# %%

from inspect import currentframe

import framework.initializer as Initializer
import framework.layer as Layer
import framework.optimizer as Optimizer
from framework.functions import *


def debug_print(arg):
    return
    frameinfo = currentframe()
    print(frameinfo.f_back.f_lineno, ":", arg)


# %%

import sys

sys.path.append("../../")

# %%

import numpy as np
from collections import OrderedDict


# %%


class MultiLayerNet:

    def __init__(self, weight_decay_lambda=0, is_use_dropout=False, dropout_ratio=0.5):

        self.weight_decay_lambda = weight_decay_lambda
        self.is_use_dropout = is_use_dropout
        self.dropout_ratio = dropout_ratio

        self.params = {}
        self.layers = OrderedDict()
        self.lastLayer = Layer.IdentityWithLoss()
        self.hiddenSizeList = []
        self.prevDenseLayer = None
        self.prevConvLayer = None

        self.pre_channel_num = None

    def reset(self):
        pass

    def add_layer(self, layer, **kwargs):
        if not isinstance(layer, Layer.LayerType):
            raise BaseException("Layer required")

        if type(self.lastLayer) == type(Layer.SoftmaxWithLoss()):
            raise BaseException("Already last layer set")

        layer_len = len(self.layers)

        if isinstance(layer, Layer.Dense):

            input_size = layer.input_size

            if input_size is None and len(self.hiddenSizeList) > 0:
                #                 input_size = self.prevDenseLayer.hidden_size
                #                 print(self.hiddenSizeList)
                input_size = int(np.prod(self.hiddenSizeList[-1]))
            else:
                self.hiddenSizeList.append(input_size)


            hidden_size = layer.hidden_size

            weight_init_std = 0.01
            initializer = layer.initializer
            if isinstance(initializer, Initializer.Std):
                weight_init_std = initializer.std
            elif isinstance(initializer, Initializer.He):
                weight_init_std = np.sqrt(2.0 / input_size)
            elif isinstance(initializer, Initializer.Xavier):
                weight_init_std = np.sqrt(1.0 / input_size)

            self.hiddenSizeList.append(hidden_size)
            #             print(input_size)
            #             print("input", self.hiddenSizeList)
            #             print(input_size, hidden_size)
            self.params[f"W{layer_len}"] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params[f"b{layer_len}"] = np.zeros(hidden_size)

            self.layers[f"Affine{layer_len}"] = Layer.Affine(self.params[f"W{layer_len}"], self.params[f"b{layer_len}"])

            self.prevDenseLayer = layer

            if layer.activation is not None:
                self.add_layer(layer.activation)

            if self.is_use_dropout:
                self.layers['Dropout' + str(layer_len)] = Layer.Dropout(self.dropout_ratio)


        elif isinstance(layer, Layer.Conv2D):

            if self.pre_channel_num == None:
                self.pre_channel_num = layer.input_size[0]

            input_size = layer.input_size

            if input_size is None and len(self.hiddenSizeList) > 0:
                input_size = self.hiddenSizeList[-1]
                print(self.hiddenSizeList)

            filter_num = layer.filter_num

            weight_init_std = 0.01
            initializer = layer.initializer
            if isinstance(initializer, Initializer.Std):
                weight_init_std = initializer.std
            elif isinstance(initializer, Initializer.He):
                weight_init_std = np.sqrt(2.0 / input_size)
            elif isinstance(initializer, Initializer.Xavier):
                weight_init_std = np.sqrt(1.0 / input_size)

            # print(layer)
            # print(layer.filter_size, layer.pad, layer.stride, input_size, "mu")

            conv_output_size = (layer.filter_num, (input_size[1] - layer.filter_size[0] + 2 * layer.pad) / layer.stride + 1,
                                (input_size[2] - layer.filter_size[1] + 2 * layer.pad) / layer.stride + 1)
            #
            self.hiddenSizeList.append(conv_output_size)

            # print("input", self.hiddenSizeList)

            debug_print(input_size)

            self.params[f"W{layer_len}"] = weight_init_std * np.random.randn(layer.filter_num, self.pre_channel_num,
                                                                             layer.filter_size[0], layer.filter_size[1])

            self.pre_channel_num = layer.filter_num

            # print("??")

            self.params[f"b{layer_len}"] = np.zeros(layer.filter_num)
            debug_print(input_size)
            self.layers[f"Convolution{layer_len}"] = Layer.Convolution(self.params[f"W{layer_len}"],
                                                                       self.params[f"b{layer_len}"], layer.stride,
                                                                       layer.pad)
            debug_print(input_size)
            self.prevConvLayer = layer

            debug_print(input_size)

            if layer.activation is not None:
                self.add_layer(layer.activation)

            if self.is_use_dropout:
                self.layers['Dropout' + str(layer_len)] = Layer.Dropout(self.dropout_ratio)
            debug_print(input_size)

        elif isinstance(layer, Layer.Pooling):
            prevLayer = self.prevConvLayer
            #             conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

            conv_output_size = self.hiddenSizeList[-1]
            print("pool", conv_output_size)
            #             print(conv_output_size)
            pool_output_size = (
                prevLayer.filter_num , int((conv_output_size[1] / layer.stride)) , int((conv_output_size[2] / layer.stride)) )
            # print("yaya", conv_output_size, pool_output_size)

            self.hiddenSizeList.append(pool_output_size)

            self.layers[f"Pooling{layer_len}"] = layer


        elif isinstance(layer, Layer.Relu):

            self.layers[f"Relu{layer_len}"] = Layer.Relu()

        elif isinstance(layer, Layer.Sigmoid):

            self.layers[f"Sigmoid{layer_len}"] = Layer.Sigmoid()

        elif isinstance(layer, Layer.SoftmaxWithLoss):

            #             self.layers[f"SoftmaxWithLoss{layer_len}"] = Layer.SoftmaxWithLoss()
            self.lastLayer = layer

        elif isinstance(layer, Layer.IdentityWithLoss):

            #             self.layers[f"SoftmaxWithLoss{layer_len}"] = Layer.SoftmaxWithLoss()
            self.lastLayer = layer


        elif isinstance(layer, Layer.BatchNormalization):

            self.layers[f"BatchNormal{layer_len}"] = Layer.BatchNormalization(
                gamma=np.ones(self.hiddenSizeList[-1]),
                beta=np.zeros(self.hiddenSizeList[-1])
            )

    def predict(self, x, train_flg=False):
        for layer in self.layers.values():

            if isinstance(layer, Layer.BatchNormalization) or isinstance(layer, Layer.Dropout):

                x = layer.forward(x, train_flg)
            else:

                x = layer.forward(x)

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx, (name, layer) in enumerate(self.layers.items()):
            if isinstance(layer, Layer.Affine):
                layer_num = name[6:]
                W = self.params['W' + str(layer_num)]
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.lastLayer.forward(y, t) + weight_decay

    def accuracy(self, x, t):

        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}

        for idx, (name, layer) in enumerate(self.layers.items()):
            if isinstance(layer, Layer.Affine):
                layer_num = name[6:]
                grads[f"W{layer_num}"] = layer.dW + self.weight_decay_lambda * self.params['W' + str(layer_num)]
                grads[f"b{layer_num}"] = layer.db
            elif isinstance(layer, Layer.Convolution):
                layer_num = name[11:]
                grads[f"W{layer_num}"] = layer.dW + self.weight_decay_lambda * self.params['W' + str(layer_num)]
                grads[f"b{layer_num}"] = layer.db

        return grads

    def train(self, x_train, t_train, x_test, t_test, **kwargs):  # t_data 는 원 핫 인코딩

        optimizer = Optimizer.SGD(lr=0.01)
        if "optimizer" in kwargs:
            if isinstance(kwargs["optimizer"], Optimizer.OptimizerType):
                optimizer = kwargs["optimizer"]
            else:
                raise "err"

        iters_num = 10000
        if "iters_num" in kwargs:
            if 0 < kwargs["iters_num"]:
                iters_num = kwargs["iters_num"]
            else:
                raise "err"

        batch_size = 100
        if "batch_size" in kwargs:
            if 0 < kwargs["batch_size"]:
                batch_size = kwargs["batch_size"]
            else:
                raise "err"

        print_epoch = 1
        if "print_epoch" in kwargs:
            if 0 < kwargs["print_epoch"]:
                print_epoch = kwargs["print_epoch"]
            else:
                raise "err"

        evaluate_limit = None
        if "evaluate_limit" in kwargs:
            if 0 < kwargs["evaluate_limit"]:
                evaluate_limit = kwargs["evaluate_limit"]
            else:
                raise "err"

        e_x_train, e_x_test, e_t_train, e_t_test = x_train, x_test, t_train, t_test

        if evaluate_limit is not None:
            e_x_train, e_x_test, e_t_train, e_t_test = x_train[:evaluate_limit], x_test[:evaluate_limit], t_train[
                                                                                                          :evaluate_limit], t_test[
                                                                                                                           :evaluate_limit]

        output_type = "regression"
        if type(self.lastLayer) == type(Layer.SoftmaxWithLoss()):
            output_type = "class"

        train_size = x_train.shape[0]

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        print(f"repeat {int(iters_num // (max(train_size / batch_size, 1)))} epoch")

        if output_type == "regression":
            print("epoch | loss")
        else:
            print("epoch | train_acc | test_acc")

        cnt = 1
        for i in range(iters_num):
            if i % 1 == 0:
                print("step", i)

            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grad = self.gradient(x_batch, t_batch)

            # 갱신

            optimizer.update(self.params, grad)

            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if int(cnt * iter_per_epoch) == i:

                if cnt % print_epoch == 0:

                    train_acc = self.accuracy(e_x_train, e_t_train)
                    test_acc = self.accuracy(e_x_test, e_t_test)
                    train_acc_list.append(train_acc)
                    test_acc_list.append(test_acc)
                    if output_type == "regression":
                        print(f"epoch {cnt}:", train_loss_list[-1])
                    else:
                        print(f"epoch {cnt}:", format(train_acc, ".4f"), " | ", format(test_acc, ".4f"))

                cnt += 1

        print("===================")
        train_acc = self.accuracy(e_x_train, e_t_train)
        test_acc = self.accuracy(e_x_test, e_t_test)
        if output_type == "regression":
            print(f"epoch {cnt - 1}:", train_loss_list[-1])
        else:
            print(f"epoch {cnt - 1}:", format(train_acc, ".4f"), " | ", format(test_acc, ".4f"))

        return {
            "train_loss_list": train_loss_list,
            "train_acc_list": train_acc_list,
            "test_acc_list": test_acc_list
        }

# %%


# %%
