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
from utils import get_model, get_cifar_gen, find_lr, get_best_checkpoint, pre_processing
from params import *
from tqdm import tqdm

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










os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


model_name = str(sys.argv[1])
threshold = float(sys.argv[2])

print(model_name, threshold)

map = {"nin": 1, "vgg19": 2, "resnet50": 3, "googlenet": 4}

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


print(model1.summary())
print(model1.evaluate(x_test, y_test))



model_layer_dict1 = defaultdict(bool)
init_dict(model1,model_layer_dict1)





layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
intermediate_layer_model = Model(inputs=model1.input,outputs=[model1.get_layer(layer_name).output for layer_name in layer_names][1:])

cov = []
flag = 0
neuronlist = []

for g in tqdm(range(len(x_test))):
    test_image = x_test[g].reshape([1,32,32,3])
    intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
    tempcount = 0
    tempstr = ''
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.max(scaled[..., num_neuron]) > threshold:
                tempcount += 1
                tempstr += '1'
                if model_layer_dict1[(layer_names[i], num_neuron)] == False:
                    model_layer_dict1[(layer_names[i], num_neuron)] = True
            else:
                tempstr += '0'
            if flag == 1:
                continue
            else:
                neuronlist.append((layer_names[i], num_neuron))
    cov.append(tempstr)
    flag = 1



tempcount = 0
totalcount = 0
for key in model_layer_dict1:
    totalcount += 1
    if model_layer_dict1[key] == True:
        tempcount += 1
print(model_layer_dict1)
print('%d / %d'%(tempcount,totalcount))

if os.path.exists(model_name + '/Cov/activeneuron/'+str(threshold)+'ase/') == False:
    os.makedirs(model_name + '/Cov/activeneuron/'+str(threshold)+'ase/')

f = open(model_name + '/Cov/activeneuron/'+str(threshold)+'ase/neuron_cov','w')
# t = open(model_name + '/Cov/activeneuron/'+str(threshold)+'ase/testinput','w')
for i in range(len(cov)):
    f.write(cov[i] + '\n')
    # t.write(str(x_test[i]) + '\n')
f.close()
# t.close()

n = open(model_name + '/Cov/activeneuron/'+str(threshold)+'ase/neuron','w')
for neuron in neuronlist:
    n.write(str(neuron) + '\n')
n.close()
