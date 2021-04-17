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

base_dir = 'D:\\google-image-crawler'

train_dir = os.path.join(base_dir, 'dataset\\train_set')

# 훈련에 사용되는 고양이/개 이미지 경로
idol_list = ["iu", "irene", "arin"]
for idol in idol_list:
    idol_dir = os.path.join(train_dir, idol)

    p = Augmentor.Pipeline(idol_dir)

    p.resize(probability=1.0, width=128, height=128)
    p.random_distortion(probability=1, grid_width=2, grid_height=2, magnitude=4)
    p.flip_left_right(probability=0.5)
    p.random_brightness(probability=0.6, min_factor=0.8, max_factor=1.2)
    p.zoom(probability=0.3, min_factor=1.05, max_factor=1.15)
    p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
    p.crop_random(probability=0.3, percentage_area=0.97)
    p.sample(10000)

