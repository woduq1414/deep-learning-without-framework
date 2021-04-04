
import sys

import matplotlib.pyplot as plt
from framework.functions import *

sys.path.append("../../")

#
import numpy as np

import pickle
with open("train_data_2021-04-04 123752","rb") as fr:
    result = pickle.load(fr)


color_list = ["red", "blue", "green", "yellow", "purple"]

marker_list = ["o", "s", "^", "v","x" ]

plt.figure(figsize=(9, 5))


def show_loss():

    plt.xlabel("step")
    plt.ylabel("loss")

    x = np.arange(len(result["train_loss_list"]))
    plt.plot(x, result["train_loss_list"], label="loss",color = color_list[0],  marker=None)

    plt.legend(loc='upper right')

    plt.show()


def show_acc():
    plt.xlabel("epoch")
    plt.ylabel("acc")

    x = np.arange(len(result["train_acc_list"]))
    plt.plot(x, result["train_acc_list"], label="train_acc", color=color_list[0], marker=None)
    plt.plot(x, result["test_acc_list"], label="test_acc", color=color_list[1], marker="o")

    for i, v in enumerate(result["test_acc_list"]):
        plt.text(i, v + 0.02, v,  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                 fontsize=9,
                 color='blue',
                 horizontalalignment='center',  # horizontalalignment (left, center, right)
                 verticalalignment='bottom')
    plt.legend(loc='upper right')

    plt.show()

#

show_acc()

