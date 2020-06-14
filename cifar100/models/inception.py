from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Activation
import sys
sys.path.append("..")
from params import *


def inception_model(input_tensor,
                    filters_1_1,
                    filters_3_3_reduce,
                    filters_3_3,
                    filters_5_5_reduce,
                    filters_5_5,
                    filters_pool_proj):
    conv_1_1 = layers.Conv2D(filters_1_1, (1, 1), padding='same')(input_tensor)
    conv_1_1 = layers.Activation('relu')(conv_1_1)

    conv_3_3_reduce = layers.Conv2D(filters_3_3_reduce, (1, 1),
                                    padding='same')(input_tensor)
    conv_3_3_reduce = layers.Activation('relu')(conv_3_3_reduce)
    conv_3_3 = layers.Conv2D(filters_3_3, (3, 3),
                             padding='same')(conv_3_3_reduce)
    conv_3_3 = layers.Activation('relu')(conv_3_3)

    conv_5_5_reduce = layers.Conv2D(filters_5_5_reduce, (1, 1),
                                    padding='same')(input_tensor)
    conv_5_5_reduce = layers.Activation('relu')(conv_5_5_reduce)
    conv_5_5 = layers.Conv2D(filters_5_5, (5, 5),
                             padding='same')(conv_5_5_reduce)
    conv_5_5 = layers.Activation('relu')(conv_5_5)

    maxpooling = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same')(input_tensor)
    maxpooling_proj = layers.Conv2D(filters_pool_proj, (1, 1),
                                    padding='same')(maxpooling)

    inception_output = layers.Concatenate()([conv_1_1, conv_3_3, conv_5_5,
                                             maxpooling_proj])

    return inception_output


def googlenet(input_shape=(img_size, img_size, channel), classes=100):
    main_input = layers.Input(input_shape)

    x = layers.Conv2D(192, (3, 3), padding='same')(main_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = inception_model(x, 64, 96, 128, 16, 32, 32)
    x = inception_model(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_model(x, 192, 96, 208, 16, 48, 64)
    x = inception_model(x, 160, 112, 224, 24, 64, 64)
    x = inception_model(x, 128, 128, 256, 24, 64, 64)
    x = inception_model(x, 112, 144, 288, 32, 64, 64)
    x = inception_model(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = inception_model(x, 256, 160, 320, 32, 128, 128)
    x = inception_model(x, 384, 192, 384, 48, 128, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.4)(x)
    # x = layers.Dense(classes, activation='softmax')(x)
    x = layers.Dense(classes)(x)
    x = Activation('softmax')(x)
    return Model(main_input, x)


if __name__ == '__main__':
    model = googlenet((img_size, img_size, channel))
    model.summary()
