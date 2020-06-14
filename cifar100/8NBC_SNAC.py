import keras
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model
from collections import defaultdict
import numpy as np
from keras import backend as K
import os,sys
import copy
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import operator
import sys
import os
import argparse
from all_output6 import get_all_output
from utils import get_model, get_cifar_gen, find_lr, get_best_checkpoint, pre_processing
from params import *
from tqdm import tqdm


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


def getSection(x_train, intermediate_layer_model,path):
    NSec = []
    for g in tqdm(range(len(x_train))):
        train_image = x_train[g].reshape([1, 32, 32, 3])
        intermediate_layer_outputs = intermediate_layer_model.predict(train_image)
        cnt = 0
        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
            # print("i:",i)
            intermediate = intermediate_layer_output[0]
            for num_neuron in range(intermediate.shape[-1]):
                # value = np.mean(intermediate[..., num_neuron])
                value = np.max(intermediate[..., num_neuron])
                if g == 0:
                    NSec.append([value,value])
                else:
                    NSec[cnt][0] = min(NSec[cnt][0], value)
                    NSec[cnt][1] = max(NSec[cnt][1], value)
                cnt += 1
    # f = open(path + 'max_min', 'w')
    # for i in tqdm(range(len(NSec))):
    #     f.write(str(NSec[i]))
    #     f.write('\n')
    # f.close()
    return NSec


def get_cov(outputs, NSec):
    tnum = len(outputs)
    nnum = len(outputs[0])
    assert nnum == len(NSec)

    Cov_NBC = []
    Cov_SNAC = []
    for j in range(tnum):
        temps_NBC = ''
        temps_SNAC = ''
        for i in range(nnum):
            tempv = outputs[j][i]
            tempmin = NSec[i][0]
            tempmax = NSec[i][1]
            if tempv < tempmin:
                temps_NBC += '10'
                temps_SNAC += '0'
            elif tempv > tempmax:
                temps_NBC += '01'
                temps_SNAC += '1'
            else:
                temps_NBC += '00'
                temps_SNAC += '0'
        Cov_NBC.append(temps_NBC)
        Cov_SNAC.append(temps_SNAC)
    return Cov_NBC, Cov_SNAC

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if ('tensorflow' == K.backend()):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    map = {"nin":1,"vgg19":2,"resnet50":3,"googlenet":4}

    model_name = str(sys.argv[1])

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
        label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # preprocess data
    x_train, x_test = pre_processing(x_train, x_test)

    model1 = load_model(str(map[model_name]) + "_" + model_name + '/' + model_name + '.h5')

    layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model1.input,
                                     outputs=[model1.get_layer(layer_name).output for layer_name in layer_names][1:])

    path1 = str(map[model_name]) + "_" + model_name + "/input/"

    NSec = getSection(x_train, intermediate_layer_model,path1)

    noutput = get_all_output(model_name)

    Cov_NBC, Cov_SNAC = get_cov(noutput, NSec)

    cov_num_NBC = cal_cov_num(Cov_NBC)
    cov_num_SNAC = cal_cov_num(Cov_SNAC)

    print('NBC:')
    ss1 = '%d / %d, %f' % (cov_num_NBC, len(Cov_NBC[0]), cov_num_NBC/(len(Cov_NBC[0])))
    print(ss1)

    f = open(path1 + 'result', 'a')
    f.write('NBC:' + ss1 + '  '+ str(model1.evaluate(x_test, y_test))+ '\n')
    f.close()

    print('SNAC:')
    ss2 = '%d / %d, %f' % (cov_num_SNAC, len(Cov_SNAC[0]) - 1 , cov_num_SNAC / (len(Cov_SNAC[0]) - 1))
    print(ss2)

    f = open(path1 + 'result', 'a')
    f.write('SNAC:' + ss2 + '  '+ str(model1.evaluate(x_test, y_test)) + '\n')
    f.close()

    f = open(path1 + '2NBC', 'w')
    for i in tqdm(range(len(Cov_NBC))):
        f.write(Cov_NBC[i] + '\n')
    f.close()

    t = open(path1 + '3SNAC', 'w')
    for i in tqdm(range(len(Cov_SNAC))):
        t.write(Cov_SNAC[i] + '\n')
    t.close()
