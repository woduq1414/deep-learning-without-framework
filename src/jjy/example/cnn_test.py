import sys
import jjy.framework.initializer as Initializer
import jjy.framework.layer as Layer
import jjy.framework.optimizer as Optimizer
from jjy.framework.functions import *
# from functions import *
from jjy.framework.network import MultiLayerNet

from jjy.dataset.mnist import load_mnist


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    print(x_train.shape, t_train.shape)
    print(t_train[0])

    net = MultiLayerNet(is_use_dropout=False)
    net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, input_size=(1, 28, 28)), initializer=Initializer.He(),
                  activation=Layer.Relu())
    # net.add_layer(Layer.Conv2D(16, (3, 3), pad=1, initializer=Initializer.He(), activation=Layer.Relu()))
    # net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    # net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, initializer=Initializer.He(), activation=Layer.Relu()))
    # net.add_layer(Layer.Conv2D(32, (3, 3), pad=2, initializer=Initializer.He(), activation=Layer.Relu()))
    # net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    # net.add_layer(Layer.Conv2D(64, (3, 3), pad=1, initializer=Initializer.He(), activation=Layer.Relu()))
    # net.add_layer(Layer.Conv2D(64, (3, 3), pad=1, initializer=Initializer.He(), activation=Layer.Relu()))
    # net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    # net.add_layer(Layer.Dense(50, initializer=Initializer.He(), activation=Layer.Relu()))
    # net.add_layer(Layer.Dropout(0.5))
    # net.add_layer(Layer.Dense(50))
    # net.add_layer(Layer.Dropout(0.5))
    # net.add_layer(Layer.SoftmaxWithLoss())

    net.add_layer(Layer.BatchNormalization())
    net.add_layer(Layer.Relu())
    # net.add_layer(Layer.Conv2D(16, (3, 3), pad=1, initializer=Initializer.He()))
    # net.add_layer(Layer.BatchNormalization())
    # net.add_layer(Layer.Relu())
    net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, initializer=Initializer.He()))
    net.add_layer(Layer.BatchNormalization())
    net.add_layer(Layer.Relu())
    # net.add_layer(Layer.Conv2D(32, (3, 3), pad=1, initializer=Initializer.He()))
    # net.add_layer(Layer.BatchNormalization())
    # net.add_layer(Layer.Relu())
    # net.add_layer(Layer.Conv2D(64, (3, 3), pad=1, initializer=Initializer.He()))
    # net.add_layer(Layer.BatchNormalization())
    # net.add_layer(Layer.Relu())
    net.add_layer(Layer.Pooling(pool_h=2, pool_w=2, stride=2))
    net.add_layer(Layer.Dense(50, initializer=Initializer.He(), activation=Layer.Relu()))
    net.add_layer(Layer.Dropout(0.5))
    net.add_layer(Layer.Dense(10))
    net.add_layer(Layer.Dropout(0.5))
    net.add_layer(Layer.SoftmaxWithLoss())



    for k, v in net.params.items():
        print(k, v.shape)

    result = net.train(
        x_train, t_train, x_test, t_test, batch_size=200, iters_num=100, print_epoch=1, evaluate_limit=500,
        is_use_progress_bar=True,
        optimizer=Optimizer.Adam(lr=0.001))

    import pickle
    import datetime
    ## Save pickle
    with open(f"train_data_{str(datetime.datetime.now())[:-7].replace(':', '')}.pickle", "wb") as fw:
        pickle.dump(result, fw)
    net.save_model()

    print("============================================")


main()

print("done!")
