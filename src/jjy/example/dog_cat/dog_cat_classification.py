import os
from PIL import Image
import jsonpickle as jsp
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from tqdm import tqdm
import jjy.framework.initializer as Initializer
import jjy.framework.layer as Layer
import jjy.framework.optimizer as Optimizer
from jjy.framework.functions import *
# from functions import *
from jjy.framework.network import MultiLayerNet

from jjy.dataset.mnist import load_mnist
import datetime


class Timer(object):

    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t = datetime.datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self.t = datetime.datetime.now() - self.t
        print(f"{self.name} Time : ", self.t)


# Open the image form working directory

def img_to_array(fname):
    image = Image.open(fname)
    image = image.resize((64, 64))
    image = np.reshape(image.convert("L"), (1, 64, 64))
    # show_img_by_array(np.asarray(image))
    return np.asarray(image)
    # return np.transpose(np.asarray(image), (2, 1, 0))


def show_img_by_array(img_array):
    transposed_array = np.transpose(img_array, (1, 2, 0))
    plt.imshow(transposed_array, cmap='gray')
    plt.show()


def get_images_from_dataset():
    base_dir = 'C:\\Users\\User\\deep-learning-without-tensorflow\\src\\jjy\\jjy\\dataset\\dog_cat'

    train_dir = os.path.join(base_dir, 'training_set\\training_set')
    test_dir = os.path.join(base_dir, 'test_set\\test_set')

    # 훈련에 사용되는 고양이/개 이미지 경로
    train_cats_dir = os.path.join(train_dir, 'cats\\output')
    train_dogs_dir = os.path.join(train_dir, 'dogs\\output')
    print(train_cats_dir)
    print(train_dogs_dir)

    # 테스트에 사용되는 고양이/개 이미지 경로
    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    print(test_cats_dir)
    print(test_dogs_dir)

    print('Total training cat images :', len(os.listdir(train_cats_dir)))
    print('Total training dog images :', len(os.listdir(train_dogs_dir)))

    print('Total validation cat images :', len(os.listdir(test_cats_dir)))
    print('Total validation dog images :', len(os.listdir(test_dogs_dir)))

    train_cat_fnames = os.listdir(train_cats_dir)
    train_dog_fnames = os.listdir(train_dogs_dir)

    test_cat_fnames = os.listdir(test_cats_dir)
    test_dog_fnames = os.listdir(test_dogs_dir)

    x_train = []

    for fname in tqdm(train_cat_fnames):
        x_train.append(img_to_array(os.path.join(train_cats_dir, fname)))

    for fname in tqdm(train_dog_fnames):
        x_train.append(img_to_array(os.path.join(train_dogs_dir, fname)))
    x_train = np.array(x_train)
    t_train = np.concatenate((np.zeros(len(train_cat_fnames)), np.ones(len(train_dog_fnames))), axis=0).astype(np.int64)

    x_test = []
    for fname in tqdm(test_cat_fnames):
        x_test.append(img_to_array(os.path.join(test_cats_dir, fname)))
    for fname in tqdm(test_dog_fnames):
        x_test.append(img_to_array(os.path.join(test_dogs_dir, fname)))
    x_test = np.array(x_test)
    t_test = np.concatenate((np.zeros(len(test_cat_fnames)), np.ones(len(test_dog_fnames))), axis=0).astype(np.int64)

    return x_train, t_train, x_test, t_test


# x_train, t_train, x_test, t_test = get_images_from_dataset()
# temp_data = {"x_train" : x_train, "x_test" : x_test, "t_train" : t_train, "t_test" : t_test}
# np.savez_compressed("./dog_cat_images_64.npz", **temp_data)
# exit(1)

def make_net1():
    net = MultiLayerNet(is_use_dropout=False)
    net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, input_size=(1, 64, 64)), initializer=Initializer.He())
    net.add_layer(Layer.BatchNormalization())
    net.add_layer(Layer.Relu())
    net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    net.add_layer(Layer.Conv2D(64, (3, 3), pad=1, initializer=Initializer.He()))
    net.add_layer(Layer.BatchNormalization())
    net.add_layer(Layer.Relu())
    net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    net.add_layer(Layer.Conv2D(128, (3, 3), pad=1, initializer=Initializer.He()))
    net.add_layer(Layer.BatchNormalization())
    net.add_layer(Layer.Relu())
    net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    net.add_layer(Layer.Dense(50, initializer=Initializer.He(), activation=Layer.Relu()))
    net.add_layer(Layer.Dropout(0.5))
    net.add_layer(Layer.Dense(2))
    net.add_layer(Layer.Dropout(0.5))
    net.add_layer(Layer.SoftmaxWithLoss())
    return net


def make_net2():
    net = MultiLayerNet(is_use_dropout=False)
    net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, input_size=(1, 64, 64)), initializer=Initializer.He(),
                  activation=Layer.Relu())
    net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    net.add_layer(Layer.Dense(128, initializer=Initializer.He(), activation=Layer.Relu()))
    net.add_layer(Layer.Dense(2, initializer=Initializer.He()))
    net.add_layer(Layer.SoftmaxWithLoss())

    return net


# def make_net2():
#     net = MultiLayerNet(is_use_dropout=False)
#     net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, input_size=(1, 64, 64)), initializer=Initializer.He(),
#                   activation=Layer.Relu())
#     net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
#     net.add_layer(Layer.Dense(128, initializer=Initializer.He(), activation=Layer.Relu()))
#     net.add_layer(Layer.Dense(2, initializer=Initializer.He()))
#     net.add_layer(Layer.SoftmaxWithLoss())
#
#     return net


def shuffle_dataset(x, t):
    from sklearn.utils import shuffle
    x, t = shuffle(x, t)
    return x, t


load_data = np.load("./dog_cat_images_64.npz", allow_pickle=True)
x_train = load_data["x_train"]

t_train = load_data["t_train"].astype(np.int64)

x_test = load_data["x_test"]
t_test = load_data["t_test"].astype(np.int64)

x_train, t_train = shuffle_dataset(x_train, t_train)
x_test, t_test = shuffle_dataset(x_test, t_test)

x_train = x_train / 255
x_test = x_test / 255

# exit(1)

#
#
def train_model():
    net = make_net1()

    result = net.train(
        x_train, t_train, x_test, t_test, batch_size=64, iters_num=1, print_epoch=1, evaluate_limit=100,
        is_use_progress_bar=True, save_model_each_epoch=1, save_model_path="./dog_cat_result",
        optimizer=Optimizer.Adam(lr=0.001))

    # print(layer_info)
    #
    # print(layer_info[-2])
    # print(layer_info)
    # exit(1)

    import pickle
    import datetime
    #
    ## Save pickle
    with open(f"train_data_{str(datetime.datetime.now())[:-7].replace(':', '')}.pickle", "wb") as fw:
        pickle.dump(result, fw)
    net.save_model("dog_cat_f2.npz")

    print("============================================")


train_model()

def load_trained_model():
    net = MultiLayerNet()

    net.load_model("dog_cat_f.npz")

    print(net.params.keys())
    return

    for k, v in net.params.items():
        print(k, v.shape)

    for i in range(5):
        img_idx = random.randrange(0, 1000)
        transposed_array = np.transpose(x_test[img_idx], (1, 2, 0))
        plt.imshow(transposed_array, cmap='gray')
        plt.show()
        predict_num = np.argmax(net.predict(np.array([x_test[img_idx]]), train_flg=False), axis=1)[0]
        print(net.predict(np.array([x_test[img_idx]])))
        correct_num = t_test[img_idx]
        print(f"Predict : {predict_num}, Correct : {correct_num}")

# load_trained_model()