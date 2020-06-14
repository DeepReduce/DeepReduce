import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.models import load_model
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys
import os
import argparse


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


from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

'''
# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
                help='dataset. (default: cifar10)')

args = parser.parse_args()

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4

#from tqdm import tqdm
'''


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = color_preprocessing(x_train, x_test)



model1 = load_model('retrain.h5')
print(model1.evaluate(x_test,y_test))
# input('check,..')
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
threshold = float(sys.argv[1])
print(threshold)
layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
intermediate_layer_model = Model(inputs=model1.input,outputs=[model1.get_layer(layer_name).output for layer_name in layer_names])

cov = []
flag = 0
neuronlist = []

for g in tqdm(range(len(x_test))):
    test_image = x_test[g].reshape([1,32,32,3])
    #print(model1.predict(test_image))
    #print(y_test[g])
    #print('*****************')
    #intermediate_layer_model = Model(inputs=model.input,outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(test_image)
    #print(intermediate_layer_outputs)
    tempcount = 0
    tempstr = ''
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            #if np.mean(scaled[..., num_neuron]) > threshold:
            if np.max(scaled[..., num_neuron]) > threshold:
                tempcount += 1
                tempstr += '1'
                if model_layer_dict1[(layer_names[i], num_neuron)] == False:
                    model_layer_dict1[(layer_names[i], num_neuron)] = True
                #    print("%s, %s : %s"%(layer_names[i], num_neuron,model_layer_dict1[(layer_names[i], num_neuron)]))
            else:
                tempstr += '0'
            if flag == 1:
                continue
            else:
                neuronlist.append((layer_names[i], num_neuron))
    cov.append(tempstr)
    flag = 1

    #print('%d : %d '%(g+1,tempcount))
    #print('*****************')

tempcount = 0
totalcount = 0
for key in model_layer_dict1:
    totalcount += 1
    if model_layer_dict1[key] == True:
        tempcount += 1
print(model_layer_dict1)
print('%d / %d'%(tempcount,totalcount))

if os.path.exists('Cov/activeneuron/'+str(threshold)+'ase/') == False:
    os.makedirs('Cov/activeneuron/'+str(threshold)+'ase/')

f = open('Cov/activeneuron/'+str(threshold)+'ase/neuron_cov','w')
# t = open('Cov/activeneuron/'+str(threshold)+'ase/testinput','w')
for i in range(len(cov)):
    f.write(cov[i] + '\n')
    # t.write(str(x_test[i]) + '\n')
f.close()
# t.close()

n = open('Cov/activeneuron/'+str(threshold)+'ase/neuron','w')
for neuron in neuronlist:
    n.write(str(neuron) + '\n')
n.close()
