import keras
from keras import optimizers
from keras.datasets import cifar10,mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
import sys
from keras import backend as K
import os
from keras.models import Model
from tqdm import tqdm


def get_all_output():
    # changelayer = int(sys.argv[1])
    # path = 'doubleneuronnumber-lenet-1/' + str(changelayer) + '/'
    path = os.getcwd() + '/'

    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_test = x_test.astype('float32')

    x_test /= 255


    y_test = keras.utils.to_categorical(y_test)

    model1 = load_model(path + 'lenet-4.h5')

    layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]

    # layer_names = [layer.name for layer in model1.layers if 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model1.input, outputs=[model1.get_layer(layer_name).output for layer_name in layer_names])

    noutput = []

    # f = open(path + 'Cov/all_neuron_values','w')

    for g in tqdm(range(len(x_test))):

        test_image = x_test[g].reshape([1,28,28,1])
        intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
        output = []

        for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):

            intermediate = intermediate_layer_output[0]
            # print(intermediate.shape)
            cur_output = []
            for num_neuron in range(intermediate.shape[-1]):
                # value = np.mean(intermediate[..., num_neuron])
                value = np.max(intermediate[..., num_neuron])
                cur_output.append(value)
            output += cur_output
        # print(len(output))
        #lenet-1 52
        noutput.append(output)
        # f.write(str(output) + '\n')

    # f.close()
    return noutput
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if('tensorflow' == K.backend()):
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    get_all_output()