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



base_dir = 'D:\\google-image-crawler\\dataset'

train_dir = os.path.join(base_dir, 'train_set')
test_dir = os.path.join(base_dir, 'test_set')

idol_list = ["iu", "irene", "arin"]

x_train = []
t_train = []

x_test = []
t_test = []


def img_to_array(fname):
    image = Image.open(fname)
    image = image.resize((128, 128))
    image = np.reshape(image.convert("L"), (1, 128, 128))
    # show_img_by_array(np.asarray(image))
    return np.asarray(image)
    # return np.transpose(np.asarray(image), (2, 1, 0))


def show_img_by_array(img_array):
    transposed_array = np.transpose(img_array, (1, 2, 0))
    plt.imshow(transposed_array, cmap='gray')
    plt.show()


for idx, idol in enumerate(idol_list):


    train_idol_dir = os.path.join(train_dir, f'{idol}\\output')
    test_idol_dir = os.path.join(test_dir, f'{idol}')

    train_idol_fname_list = os.listdir(train_idol_dir)
    test_idol_fname_list = os.listdir(test_idol_dir)

    for fname in tqdm(train_idol_fname_list):
        x_train.append(img_to_array(os.path.join(train_idol_dir, fname)))
        t_train.append(idx)


    for fname in tqdm(test_idol_fname_list):
        x_test.append(img_to_array(os.path.join(test_idol_dir, fname)))
        t_test.append(idx)

x_train = np.array(x_train)
x_test = np.array(x_test)
t_train = np.array(t_train)
t_test = np.array(t_test)

print(x_train.shape)
temp_data = {"x_train" : x_train, "x_test" : x_test, "t_train" : t_train, "t_test" : t_test}
np.savez_compressed("./idol_images_gray_128.npz", **temp_data)