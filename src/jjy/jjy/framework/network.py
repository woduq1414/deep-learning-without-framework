# %%
import jsonpickle as jsp
from inspect import currentframe

import jjy.framework.initializer as Initializer
import jjy.framework.layer as Layer
import jjy.framework.optimizer as Optimizer
from jjy.framework.functions import *
from tqdm import tqdm
import os
import json


def debug_print(arg):
    return
    frameinfo = currentframe()
    print(frameinfo.f_back.f_lineno, ":", arg)


import datetime
# %%

import sys

sys.path.append("../../")

# %%
import re
import numpy as np
from collections import OrderedDict


# %%


class MultiLayerNet:

    def __init__(self, weight_decay_lambda=0, is_use_dropout=False, dropout_ratio=0.5):

        self.is_gpu = False

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

        self.added_layer_list = []

    def to_gpu(self):
        self.is_gpu = True

    def reset(self):
        pass

    def add_layer(self, layer, **kwargs):

        is_direct = True
        if "is_direct" in kwargs and kwargs["is_direct"] == False:
            is_direct = False

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

            pre_node_nums = input_size

            if isinstance(initializer, Initializer.Std):
                weight_init_std = initializer.std
            elif isinstance(initializer, Initializer.He):
                weight_init_std = np.sqrt(2.0 / pre_node_nums)
            elif isinstance(initializer, Initializer.Xavier):
                weight_init_std = np.sqrt(1.0 / pre_node_nums)

            self.hiddenSizeList.append(hidden_size)
            #             print(input_size)
            #             print("input", self.hiddenSizeList)
            #             print(input_size, hidden_size)
            self.params[f"W{layer_len}"] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params[f"b{layer_len}"] = np.zeros(hidden_size)

            self.layers[f"Affine{layer_len}"] = Layer.Affine(self.params[f"W{layer_len}"], self.params[f"b{layer_len}"])

            self.prevDenseLayer = layer

            if layer.activation is not None:
                self.add_layer(layer.activation, is_direct=False)

            if self.is_use_dropout:
                layer_len = len(self.layers)
                self.layers['Dropout' + str(layer_len)] = Layer.Dropout(self.dropout_ratio)


        elif isinstance(layer, Layer.Conv2D):

            if self.pre_channel_num == None:
                self.pre_channel_num = layer.input_size[0]

            input_size = layer.input_size

            if input_size is None and len(self.hiddenSizeList) > 0:
                input_size = self.hiddenSizeList[-1]
                # print(input_size)
                # print(self.hiddenSizeList)

            filter_num = layer.filter_num

            weight_init_std = 0.01
            initializer = layer.initializer

            if self.prevConvLayer is None:
                pre_node_nums = np.prod(layer.filter_size)
            else:
                pre_node_nums = self.prevConvLayer.filter_num * np.prod(layer.filter_size)

            if isinstance(initializer, Initializer.Std):
                weight_init_std = initializer.std
            elif isinstance(initializer, Initializer.He):
                weight_init_std = np.sqrt(2.0 / pre_node_nums)
            elif isinstance(initializer, Initializer.Xavier):
                weight_init_std = np.sqrt(1.0 / pre_node_nums)

            # print(pre_node_nums)

            # print(layer)
            # print(layer.filter_size, layer.pad, layer.stride, input_size, "mu")

            conv_output_size = (
                layer.filter_num, int((input_size[1] - layer.filter_size[0] + 2 * layer.pad) / layer.stride + 1),
                int((input_size[2] - layer.filter_size[1] + 2 * layer.pad) / layer.stride + 1))
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
                self.add_layer(layer.activation, is_direct=False)

            if self.is_use_dropout:
                self.layers['Dropout' + str(layer_len)] = Layer.Dropout(self.dropout_ratio)
            debug_print(input_size)

        elif isinstance(layer, Layer.Pooling):
            prevLayer = self.prevConvLayer
            #             conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1

            conv_output_size = self.hiddenSizeList[-1]

            #             print(conv_output_size)
            pool_output_size = (
                prevLayer.filter_num, int((conv_output_size[1] / layer.stride)),
                int((conv_output_size[2] / layer.stride)))
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
            # print(self.hiddenSizeList)
            self.layers[f"BatchNormal{layer_len}"] = Layer.BatchNormalization(
                gamma=np.ones(np.prod(self.hiddenSizeList[-1])),
                beta=np.zeros(np.prod(self.hiddenSizeList[-1])),
                running_mean=layer.running_mean,
                running_var=layer.running_var
            )
        # print(layer, self.hiddenSizeList)
        if is_direct:
            self.added_layer_list.append(layer)

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

        iters_num = kwargs.get("iters_num", None)
        if iters_num is None:
            return ValueError("iters_num must be set")
        elif not (iters_num > 0):
            raise ValueError("iters_num must be > 0 ")

        batch_size = kwargs.get("batch_size", None)
        if batch_size is None:
            return ValueError("batch_size must be set")
        elif not (batch_size > 0):
            raise ValueError("batch_size must be > 0")

        print_epoch = kwargs.get("print_epoch", 1)
        if print_epoch is None:
            print_epoch = 9999999999999999999
        elif not (print_epoch > 0):
            raise ValueError("print_epoch must be > 0")

        save_model_each_epoch = kwargs.get("save_model_each_epoch", 1)
        if save_model_each_epoch is None:
            save_model_each_epoch = 9999999999999999999
        elif not (save_model_each_epoch > 0):
            raise ValueError("save_model_each_epoch must be > 0")

        evaluate_limit = kwargs.get("evaluate_limit", None)
        save_model_path = kwargs.get("save_model_path", "")

        is_use_progress_bar = kwargs.get("is_use_progress_bar", False)

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

        if is_use_progress_bar:
            iterator = tqdm(range(iters_num), mininterval=1)
        else:
            iterator = range(iters_num)

        for i in iterator:
            # with nostdout():
            # with redirect_to_tqdm():

            # print(i)

            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grad = self.gradient(x_batch, t_batch)

            # 갱신

            optimizer.update(self.params, grad)

            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if int(cnt * iter_per_epoch) == i:

                if save_model_each_epoch is not None and cnt % save_model_each_epoch == 0:
                    self.save_model(
                        save_model_path + f"/train_weight_{str(datetime.datetime.now())[:-7].replace(':', '')}.npz")

                if cnt % print_epoch == 0:

                    train_acc = self.accuracy(e_x_train, e_t_train)
                    test_acc = self.accuracy(e_x_test, e_t_test)
                    train_acc_list.append(train_acc)
                    test_acc_list.append(test_acc)
                    if output_type == "regression":
                        print(f"epoch {cnt}:", train_loss_list[-1])
                    else:
                        # log.info(f"epoch {cnt}:")
                        print(f"epoch {cnt}:", format(train_acc, ".4f"), " | ", format(test_acc, ".4f"), flush=True)

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

    def save_model(self, path=None):
        save_data = OrderedDict()
        params = self.params

        save_data["NetConfig"] = {
            "weight_decay_lambda": self.weight_decay_lambda,
            "is_use_dropout": self.is_use_dropout,
            "dropout_ratio": self.dropout_ratio
        }
        layer_info = []
        for layer in self.added_layer_list:
            t = layer
            if isinstance(t, Layer.SoftmaxWithLoss):
                t = Layer.SoftmaxWithLoss()
            elif isinstance(t, Layer.IdentityWithLoss):
                t = Layer.IdentityWithLoss()
            elif isinstance(t, Layer.BatchNormalization):
                t = Layer.BatchNormalization(running_mean=t.running_mean, running_var=t.running_var)
            elif isinstance(t, Layer.Pooling):
                t = Layer.Pooling(t.pool_h, t.pool_w, t.stride, t.pad)

            s = jsp.encode(t)
            layer_info.append(s)

        save_data["LayerInfo"] = layer_info
        # for layer_idx, layer in enumerate(self.layers.values()):
        #     if isinstance(layer, Layer.BatchNormalization):
        #         params[f"BN_m{layer_idx}"] = layer.running_mean
        #         params[f"BN_v{layer_idx}"] = layer.running_var

        save_data["Params"] = params

        if path is None:
            path = f"train_weight_{str(datetime.datetime.now())[:-7].replace(':', '')}.npz"
        # print(save_data)


        np.savez_compressed(path, **save_data)
        print(f"Weight was saved at {path}")

    def load_model(self, path):

        print(f"Loading Model... {path}")
        load_data = np.load(path, allow_pickle=True)
        other_config_data = dict(load_data.get("NetConfig").item())
        self.__init__(**other_config_data)
        layer_info = load_data.get("LayerInfo")
        for layer_encoded in layer_info:
            layer = jsp.decode(layer_encoded)
            self.add_layer(layer)

        self.params = load_data.get("Params").item()

        for param_name, param_value in self.params.items():
            print(param_name)

            layer_idx = int(re.findall(r'[0-9]+', param_name)[0])
            param_kind = re.sub(r'[0-9]+', '', param_name)
            target_layer = list(self.layers.items())[layer_idx][1]
            if param_kind == "W":
                target_layer.W = param_value
            elif param_kind == "b":
                target_layer.b = param_value
            elif param_kind == "BN_m":
                target_layer.running_mean = param_value
            elif param_kind == "BN_v":
                target_layer.running_var = param_value

# %%


# %%
