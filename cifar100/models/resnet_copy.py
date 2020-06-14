from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.regularizers import l2

import sys
sys.path.append("..")
from params import *


def basic_block(input_tensor, filters, stride=1, weight_decay=5e-4):
    expansion = 1
    x = layers.Conv2D(filters, (3, 3),
                      strides=stride,
                      padding='same',
                      kernel_regularizer=l2(weight_decay),
                      kernel_initializer='he_normal',
                      use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters * expansion, (3, 3),
                      padding='same',
                      kernel_regularizer=l2(weight_decay),
                      kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    shortcut = input_tensor
    # if stride != 1 then the w h of feature map is reduced
    # if input_tensor.shape[-1] != filters * expansion then the dimension needed to be improved
    if stride != 1 or input_tensor.shape[-1] != filters * expansion:
        shortcut = layers.Conv2D(filters * expansion, (1, 1),
                                 strides=stride,
                                 padding='valid',
                                 kernel_regularizer=l2(weight_decay),
                                 kernel_initializer='he_normal',
                                 use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def bottleneck_block(input_tensor, filters, stride=1, weight_decay=5e-4):
    expansion = 4
    x = layers.Conv2D(filters, (1, 1),
                      padding='valid',
                      kernel_regularizer=l2(weight_decay),
                      kernel_initializer='he_normal',
                      use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3),
                      strides=stride,
                      padding='same',
                      kernel_regularizer=l2(weight_decay),
                      kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters * expansion, (1, 1),
                      padding='valid',
                      kernel_regularizer=l2(weight_decay),
                      kernel_initializer='he_normal',
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    shortcut = input_tensor
    if stride != 1 or input_tensor.shape[-1] != filters * expansion:
        shortcut = layers.Conv2D(filters * expansion, (1, 1),
                                 strides=stride,
                                 padding='valid',
                                 kernel_regularizer=l2(weight_decay),
                                 kernel_initializer='he_normal',
                                 use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def resnet(block, num_block,
           input_shape=(img_size, img_size, channel),
           weight_decay=5e-4):
    main_input = layers.Input(input_shape)

    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_regularizer=l2(weight_decay),
                      kernel_initializer='he_normal',
                      use_bias=False)(main_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = make_layer(x, block, 64, num_block[0], 1)
    x = make_layer(x, block, 128, num_block[1], 2)
    x = make_layer(x, block, 256, num_block[2], 2)
    x = make_layer(x, block, 512, num_block[3], 2)
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(num_classes, activation='softmax')(x)
    x = layers.Dense(num_classes)(x)
    x = Activation('softmax')(x)

    return Model(main_input, x)


def make_layer(input_tensor, block, filters, num_blocks, stride, weight_decay=5e-4):

    x = block(input_tensor, filters, stride, weight_decay)
    for _ in range(num_blocks - 1):
        x = block(x, filters, 1, weight_decay)

    return x


def resnet18():
    return resnet(basic_block, [2, 2, 2, 2])


def resnet34():
    return resnet(basic_block, [3, 4, 6, 3])


def resnet50():
    return resnet(bottleneck_block, [3, 4, 6, 3])


def resnet101():
    return resnet(bottleneck_block, [3, 4, 23, 3])


def resnet152():
    return resnet(bottleneck_block, [3, 8, 36, 3])


if __name__ == '__main__':
    model = resnet34()
    model.summary()
