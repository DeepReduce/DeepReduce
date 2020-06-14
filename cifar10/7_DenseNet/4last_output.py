import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np

def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def update_coverage(input_data, model, model_layer_dict, threshold=0.2):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in xrange(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # mean = [125.307, 122.95, 113.865]
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test


from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = color_preprocessing(x_train, x_test)

model1 = load_model('densenet.h5')
print(model1.evaluate(x_test, y_test))
#print(model1.summary())
#input('check...')
#for layer in model1.layers:
    #for index in range(layer.output_shape[-1]):
    #    print(layer.name)
    #    print(layer.output_shape)
    #print(layer.name)
    #print(layer.output_shape)
    #print(layer.output_shape[-1])
    #print('----------')

model_layer_dict1 = defaultdict(bool)
init_dict(model1,model_layer_dict1)
#print(model_layer_dict1)
#print(len(model_layer_dict1.keys()))
#test_image = x_test[0].reshape([1,32,32,3])
#test_image.shape
#res = model.predict(test_image)
#label = softmax_to_label(res)
#print(label)
#print(x_test[0])
#print(len(x_test[0]))
#print(len(x_test[0][0]))
from keras.models import Model

#threshold = float(0.5)
layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
#print(layer_names)
#input('check...')
#intermediate_layer_model = Model(inputs=model1.input,outputs=[model1.get_layer(layer_name).output for layer_name in layer_names])
intermediate_layer_model = Model(inputs=model1.input, outputs = [model1.get_layer(layer_names[-2]).output])

from tqdm import tqdm

cov = []
flag = 0
neuronlist = []

f = open('last-hidden','w')

for g in tqdm(range(len(x_test))):
    test_image = x_test[g].reshape([1,32,32,3])

    intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
    #print(type(intermediate_layer_outputs[0]))
    #print(intermediate_layer_outputs[0])
    output = intermediate_layer_outputs[0].tolist()
    #print(output)
    #print(intermediate_layer_output[0])
    #print(len(intermediate_layer_output[0]))
    #input('pause...')
    f.write(str(output) + '\n')
f.close()
