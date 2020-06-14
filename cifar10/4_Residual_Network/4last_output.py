import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
from keras import backend as K
import os
from keras.models import Model
from tqdm import tqdm

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test


def cal(index,ss):

    t = open('Cov/layername','a')
    t.write(str(index) + layer_names[index] + '\n')
    t.close()

    intermediate_layer_model = Model(inputs=model1.input, outputs=[model1.get_layer(layer_names[index]).output])


    cov = []
    flag = 0
    neuronlist = []

    f = open('Cov/' + ss, 'w')

    for g in tqdm(range(len(x_test))):
        test_image = x_test[g].reshape([1, 32, 32, 3])

        intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
        # print(intermediate_layer_outputs.shape)
        intermediate_layer_output = intermediate_layer_outputs[0]
        # print(intermediate_layer_output.shape)
        output = []
        for num_neuron in range(intermediate_layer_output.shape[-1]):
            output.append(np.max(intermediate_layer_output[..., num_neuron]))
        # output = intermediate_layer_outputs[0].tolist()
        f.write(str(output) + '\n')
    f.close()


if __name__ == '__main__':

    if ('tensorflow' == K.backend()):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = color_preprocessing(x_train, x_test)

    model1 = load_model('resnet_32_cifar10_new.h5')
    print(model1.summary())
    print('acc: ')
    print(model1.evaluate(x_test,y_test))


    layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    t = open('Cov/layername', 'a')
    t.write(str(layer_names) + '\n')
    t.close()
    cal(-1, 'last-layer')
    cal(-2,'last-output')
    cal(-3,'last-hidden')










