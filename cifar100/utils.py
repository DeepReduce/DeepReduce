import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from params import *
from models import resnet
from models import resnet_copy
from models.vgg import vgg11, vgg13, vgg16, vgg19
from models.nin import nin
from models.inception import googlenet


def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels


def pre_processing(train_images, test_images):
    images = np.concatenate((train_images, test_images), axis = 0)
    mean, std = get_mean_std(images)

    for i in range(train_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images


def get_cifar_gen():
    # get dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
        label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    # preprocess data
    x_train, x_test = pre_processing(x_train, x_test)
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    cifar_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

    testgen = ImageDataGenerator()
    cifar_test_gen = testgen.flow(x_test, y_test, batch_size=batch_size)

    return cifar_gen, cifar_test_gen


def get_model(train_model):

    if train_model == 'resnet18':
        return resnet.resnet18()
    elif train_model == 'resnet34':
        return resnet.resnet34()
    elif train_model == 'resnet50':
        return resnet.resnet50()
    elif train_model == 'resnet101':
        return resnet.resnet101()
    elif train_model == 'resnet152':
        return resnet.resnet152()
    elif train_model == 'resnet18_copy':
        return resnet_copy.resnet18()
    elif train_model == 'resnet34_copy':
        return resnet_copy.resnet34()
    elif train_model == 'resnet50_copy':
        return resnet_copy.resnet50()
    elif train_model == 'resnet101_copy':
        return resnet_copy.resnet101()
    elif train_model == 'resnet152':
        return resnet_copy.resnet152()
    elif train_model == 'vgg11':
        return vgg11()
    elif train_model == 'vgg13':
        return vgg13()
    elif train_model == 'vgg16':
        return vgg16()
    elif train_model == 'vgg19':
        return vgg19()
    elif train_model == 'nin':
        return nin()
    elif train_model == 'googlenet':
        return googlenet()


def find_lr(epoch_idx, cur_lr):
    if epoch_idx < 1:  # warm up
        return cur_lr / epochs
    if epoch_idx < 60:
        return 0.1
    elif epoch_idx < 120:
        return 0.02
    elif epoch_idx < 160:
        return 0.004
    else:
        return 0.0008


def get_best_checkpoint(dir_name):
    cpt_path = os.path.join('experiments', dir_name, 'checkpoints')
    if os.path.exists(cpt_path):
        all_cpt = os.listdir(cpt_path)
        if len(all_cpt) > 0:
            best_val_loss = 100
            best_idx = 0
            for idx, cpt in enumerate(all_cpt):
                val_loss = cpt[-11: -5]
                if float(val_loss) < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_idx = idx
            return all_cpt[best_idx]
    return None


def save_train_images(history, save_path):
    # plot loss and acc
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('Loss value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('acc value')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(save_path, 'acc.png'))
    plt.close()
