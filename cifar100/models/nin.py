from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Activation

import sys
sys.path.append("..")
from params import *


def nin_block(input_tensor, kernel_size, filters, strides=(1, 1), single=False):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1, 1), padding='same')(x)
    x = layers.Activation('relu')(x)
    if not single:
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.Activation('relu')(x)
    return x


def nin(input_shape=(img_size, img_size, channel), classes=100, single=False):
    main_input = layers.Input(input_shape)
    x = nin_block(main_input, (3, 3), 64, single=single)
    x = layers.MaxPooling2D()(x)
    x = nin_block(x, (3, 3), 128, single=single)
    x = layers.MaxPooling2D()(x)
    x = nin_block(x, (3, 3), 256, single=single)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(classes, activation='softmax')(x)
    x = layers.Dense(classes)(x)
    x = Activation('softmax')(x)

    return Model(main_input, x)


if __name__ == '__main__':
    model = nin((img_size, img_size, channel))
    model.summary()
