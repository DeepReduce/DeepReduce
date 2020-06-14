import keras
from keras import optimizers
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
import sys
import os

from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


if __name__ == '__main__':
    changelayer = int(sys.argv[1])
    path = 'halveneuronnumber-lenet-5/' + str(changelayer) + '/'

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 处理 x
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_test = x_test.astype('float32')

    x_test /= 255

    # 处理 y
    y_test = keras.utils.to_categorical(y_test)

    model1 = load_model(path + 'lenet-5.h5')
    # print(model.summary())
    print(model1.evaluate(x_test, y_test))
    # input('check...')
    count = 0
    acc = 0
    predict = []
    for i in range(len(x_test)):
        test_image = x_test[i].reshape([1, 28, 28, 1])
        y = model1.predict(test_image)
        y_label = np.argmax(y)
        if y_label == np.argmax(y_test[i]):
            acc += 1
            predict.append((1, y_label, np.argmax(y_test[i])))
        else:
            predict.append((0, y_label, np.argmax(y_test[i])))
        count += 1
        # print('%s - %s'%(y,np.argmax(y_test[i])))
    print(model1.evaluate(x_test, y_test))
    print("total : %s, acc : %s, accratio : %s" % (count, acc, acc / (count * 1.0)))
    f = open(path + 'Cov/predict', 'w')
    for i in range(len(x_test)):
        f.write(str(predict[i]) + '\n')
    f.close()


