
import keras
from keras import optimizers
from keras.datasets import cifar10, mnist
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
        test_image = x_test[g].reshape([1, 28, 28, 1])

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 处理 x
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_test = x_test.astype('float32')

    x_test /= 255

    # 处理 y
    y_test = keras.utils.to_categorical(y_test)

    model1 = load_model('lenet-4.h5')
    print(model1.summary())
    print('acc: ')
    print(model1.evaluate(x_test,y_test))


    layer_names = [layer.name for layer in model1.layers if  'input' not in layer.name]
    t = open('Cov/layername', 'a')
    t.write(str(layer_names) + '\n')
    t.close()
    cal(-1,'last_layer')
    cal(-2,'last_output')
    cal(-3,'last_hidden')


