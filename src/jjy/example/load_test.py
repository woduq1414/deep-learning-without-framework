from framework.network import MultiLayerNet
import random
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

net = MultiLayerNet()
net.load_model("train_weight_2021-04-10 213538.npz")

for i in range(5):
    img_idx = random.randrange(0, 10000)

    plt.imshow(x_test[img_idx].reshape(28, 28), cmap='gray')
    plt.show()
    predict_num = np.argmax(net.predict(np.array([x_test[img_idx]]), train_flg=False), axis=1)[0]
    correct_num = t_test[img_idx]
    print(f"Predict : {predict_num}, Correct : {correct_num}")

