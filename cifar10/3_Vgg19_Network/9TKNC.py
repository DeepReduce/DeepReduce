
import keras
from keras import optimizers
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from keras.models import Model
import copy
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys
import os
from keras import backend as K
import argparse
from all_output6 import get_all_output

def readFile(filepath):
    f = open(filepath)
    content = f.read()
    f.close()
    return content.splitlines()


def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # mean = [125.307, 122.95, 113.865]
    mean = [123.680, 116.779, 103.939]
    # std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i])
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i])

    return x_train, x_test




def getCoverage(x_test,intermediate_layer_model,k):

    Top_k_Neuron_Coverage = []
    print("get Top k Neuron Coverage...")
    for g in tqdm(range(len(x_test))):
        train_image = x_test[g].reshape([1, 32, 32, 3])
        intermediate_layer_outputs = intermediate_layer_model.predict(train_image)
        cur_coverage = ''
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            intermediate = intermediate_layer_output[0]
            cur_layer = []
            cur_cov = [0] * intermediate.shape[-1]
            cur_str = ''
            for num_neuron in range(intermediate.shape[-1]):
                # value = np.mean(intermediate[..., num_neuron])
                value = np.max(intermediate[..., num_neuron])
                cur_layer.append(value)
            indexes = np.argpartition(cur_layer, -k)[-k:]
            # indexes = np.argmax(cur_layer)
            for index in indexes:
                cur_cov[index] = 1
            for i in range(intermediate.shape[-1]):
                cur_str += str(cur_cov[i])
            cur_coverage += cur_str
        Top_k_Neuron_Coverage.append(cur_coverage)
    return Top_k_Neuron_Coverage

def cal_cov_num(Cov):
    tnum = len(Cov)
    nnum = len(Cov[0])
    cov_num = 0
    for i in range(nnum):
        for j in range(tnum):
            if Cov[j][i] == '1':
                cov_num += 1
                break
    return cov_num

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if ('tensorflow' == K.backend()):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    k = int(sys.argv[1])

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train, x_test = color_preprocessing(x_train, x_test)


    filename = 'retrain.h5'
    model1 = load_model(filename)
    print(model1.evaluate(x_test, y_test))

    layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model1.input,
                                     outputs=[model1.get_layer(layer_name).output for layer_name in layer_names])

    Top_k_Neuron_Coverage = getCoverage(x_test,intermediate_layer_model,k)

    print("write to file...")
    f = open('Cov/4TKNC_' + str(k), 'w')
    for i in tqdm(range(len(Top_k_Neuron_Coverage))):
        f.write(str(Top_k_Neuron_Coverage[i]) + '\n')
    f.close()

    total_neuron = len(Top_k_Neuron_Coverage[0])
    top_k_neuron = cal_cov_num(Top_k_Neuron_Coverage)

    ss = '%d / %d %f'%(top_k_neuron,total_neuron,top_k_neuron/total_neuron)
    print(ss)

    f = open('Cov/result', 'a')
    f.write(str(model1.evaluate(x_test, y_test)) + '4TKNC_' + str(k) + ':' + ss + '\n')
    f.close()

