import os
from PIL import Image
import Augmentor
import matplotlib.pyplot as plt
import numpy as np
import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import rotate



def show_img(img, ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img)


def plot_grid(imgs, nrows, ncols, figsize=(10, 10)):
    assert len(imgs) == nrows * ncols, f"Number of images should be {nrows}x{ncols}"
    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        show_img(img, ax)


#

base_dir = '/jjy/dataset/dog_cat'

train_dir = os.path.join(base_dir, 'training_set\\training_set')
test_dir = os.path.join(base_dir, 'test_set\\test_set')

# 훈련에 사용되는 고양이/개 이미지 경로
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

p = Augmentor.Pipeline(train_cats_dir)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
p.flip_left_right(probability=0.5)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.25)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.crop_random(probability=0.5, percentage_area=0.9)
p.sample(10000)

