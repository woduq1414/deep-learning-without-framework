
import sys

import matplotlib.pyplot as plt
from jjy.framework.functions import *
from PIL import Image
import os
#
import jjy.framework.layer as Layer
import jjy.framework.optimizer as Optimizer
import jjy.framework.initializer as Initializer
from jjy.framework.functions import *
from jjy.framework.network import MultiLayerNet
import random

import numpy as np

import pickle
with open("train_data_2021-04-17 061246.pickle","rb") as fr:
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
        plt.text(i, v + 0.0003, v,  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                 fontsize=9,
                 color='blue',
                 horizontalalignment='center',  # horizontalalignment (left, center, right)
                 verticalalignment='bottom')
    plt.legend(loc='lower right')

    plt.show()

def show_img(img,t, ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(t, fontsize=20)


def plot_grid(imgs, title_list, nrows, ncols, figsize=(10, 10)):



    assert len(imgs) == nrows * ncols, f"Number of images should be {nrows}x{ncols}"
    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()
    for i, (img, ax) in enumerate(zip(imgs, axs)):
        show_img(img,title_list[i], ax)


def show_img_predict():
    net = MultiLayerNet()
    net.load_model("dog_cat_f.npz")

    print(net.layers)
    print(net.params)


    base_dir = 'C:\\Users\\User\\deep-learning-without-tensorflow\\src\\jjy\\jjy\\dataset\\dog_cat'

    test_dir = os.path.join(base_dir, 'test_set\\test_set')

    # 테스트에 사용되는 고양이/개 이미지 경로
    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    print(test_cats_dir)
    print(test_dogs_dir)

    print('Total validation cat images :', len(os.listdir(test_cats_dir)))
    print('Total validation dog images :', len(os.listdir(test_dogs_dir)))

    test_cat_fnames = os.listdir(test_cats_dir)
    test_dog_fnames = os.listdir(test_dogs_dir)

    def img_to_array(fname, original=False):

        image = Image.open(fname)

        if original is False:
            image = image.resize((64, 64))
            image = np.reshape(image.convert("L"), (1, 64, 64))
        # show_img_by_array(np.asarray(image))
        return np.asarray(image)

    random.shuffle(test_cat_fnames)
    random.shuffle(test_dog_fnames)

    img_list = []
    predict_list = []
    for fname in test_dog_fnames[:15]:

        fname = os.path.join(test_dogs_dir, fname)

        img_original_array = img_to_array(fname, original=True)
        img_array = img_to_array(fname)
        img_array = img_array / 255.0

        img_list.append(img_original_array)
        predict_num = np.argmax(net.predict(np.array([img_array]), train_flg=False), axis=1)[0]
        print(net.predict(np.array([img_array]), train_flg=False))
        # print(predict_num)

        predict_list.append("CAT!" if predict_num == 0 else "DOG!")

    plot_grid(img_list, predict_list, 3, 5)
    plt.show()




show_img_predict()

# show_loss()
# show_acc()

