from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.regularizers import l2

import sys
sys.path.append("..")
from params import *

vgg_cfg = {
    11: [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg(layer_cfg,
        input_shape=(img_size, img_size, channel),
        weight_decay=5e-4,
        use_bias=False,
        use_fc=False):

    block_num = 1
    conv_num = 1
    x = layers.Input(input_shape, name='vgg16_bn')
    main_input = x

    for layer in layer_cfg:
        if layer == 'M':
            x = layers.MaxPooling2D((2, 2), strides=(2, 2),
                                    name='block%d_pool' % block_num)(x)
            block_num += 1
            conv_num = 1
            continue

        x = layers.Conv2D(layer, (3, 3),
                          padding='same',
                          name='block%d_conv%d' % (block_num, conv_num),
                          kernel_regularizer=l2(weight_decay),
                          use_bias=use_bias)(x)
        x = layers.BatchNormalization(name='block%d_bn%d' % (block_num, conv_num))(x)
        x = layers.Activation('relu', name='block%d_relu%d' % (block_num, conv_num))(x)
        conv_num += 1

    if use_fc:
        x = layers.Dense(4096)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        x = layers.Dense(4096)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(rate=0.5)(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(num_classes, activation='softmax')(x)
    x = layers.Dense(num_classes)(x)
    x = Activation('softmax')(x)
    return Model(main_input, x)


def vgg11():
    return vgg(layer_cfg=vgg_cfg[11])


def vgg13():
    return vgg(layer_cfg=vgg_cfg[13])


def vgg16():
    return vgg(layer_cfg=vgg_cfg[16])


def vgg19():
    return vgg(layer_cfg=vgg_cfg[19])


if __name__ == '__main__':
    model = vgg19()
    model.summary()
