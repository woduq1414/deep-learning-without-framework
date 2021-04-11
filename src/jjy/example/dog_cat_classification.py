import os

base_dir = 'C:\\Users\\User\\deep-learning-without-tensorflow\\src\\jjy\\jjy\\dataset\\dog_cat'

train_dir = os.path.join(base_dir, 'training_set\\training_set')
validation_dir = os.path.join(base_dir, 'test_set\\test_set')

# 훈련에 사용되는 고양이/개 이미지 경로
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
print(train_cats_dir)
print(train_dogs_dir)

# 테스트에 사용되는 고양이/개 이미지 경로
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print(validation_cats_dir)
print(validation_dogs_dir)

print('Total training cat images :', len(os.listdir(train_cats_dir)))
print('Total training dog images :', len(os.listdir(train_dogs_dir)))

print('Total validation cat images :', len(os.listdir(validation_cats_dir)))
print('Total validation dog images :', len(os.listdir(validation_dogs_dir)))

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

def show_images():
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    nrows, ncols = 4, 4
    pic_index = 0

    fig = plt.gcf()
    fig.set_size_inches(ncols * 3, nrows * 3)

    pic_index += 8

    next_cat_pix = [os.path.join(train_cats_dir, fname)
                    for fname in train_cat_fnames[pic_index - 8:pic_index]]

    next_dog_pix = [os.path.join(train_dogs_dir, fname)
                    for fname in train_dog_fnames[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()