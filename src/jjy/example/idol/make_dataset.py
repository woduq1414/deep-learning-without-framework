import os
import shutil
import random

base_dir = 'D:\\google-image-crawler'

crop_dir = os.path.join(base_dir, 'crop')

idol_list = ["iu", "irene", "arin"]
for idol in idol_list:

    idol_dir = os.path.join(crop_dir, idol)
    fname_list = os.listdir(idol_dir)

    random.shuffle(fname_list)

    for fname in fname_list[:1000]:
        shutil.copy(os.path.join(idol_dir, fname), f"D:\\google-image-crawler\\dataset\\train_set\\{idol}")
    for fname in fname_list[1000:]:
        shutil.copy(os.path.join(idol_dir, fname), f"D:\\google-image-crawler\\dataset\\test_set\\{idol}")


