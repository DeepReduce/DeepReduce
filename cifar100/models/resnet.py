from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import Model
from tensorflow.python.keras.regularizers import l2

import sys
sys.path.append("..")
from params import *


def identity_block(input_tensor, filters, stage, block, weight_decay=5e-4, use_bias=False):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        filters: the filters of the first conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters * 4, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               filters,
               stage,
               block,
               strides=(2, 2),
               weight_decay=5e-4,
               use_bias=False):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        filters: the filters of the first conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters * 4, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters * 4, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',
                             kernel_regularizer=l2(weight_decay),
                             use_bias=use_bias)(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block_basic(input_tensor, filters, stage, block, weight_decay=5e-4, use_bias=False):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_basic(input_tensor,
                     filters,
                     stage,
                     block,
                     strides=(2, 2),
                     weight_decay=5e-4,
                     use_bias=False):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters, (3, 3), strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    shortcut = layers.Conv2D(filters, (1, 1), strides=strides,
                             padding='valid',
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',
                             kernel_regularizer=l2(weight_decay),
                             use_bias=use_bias)(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet(num_block,
           input_shape=(img_size, img_size, channel),
           classes=100,
           weight_decay=5e-4,
           use_bias=False,
           bottleneck=True):
    main_input = layers.Input(input_shape, name='resnet_input')
   
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      kernel_initializer='he_normal',
                      name='conv1',
                      kernel_regularizer=l2(weight_decay),
                      use_bias=use_bias)(main_input)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)

    ascii_a = 97
    # for resnet50, 101, 152, bottleneck block is used
    # for resnet18, 34, basic block is used
    if bottleneck:
        net_conv_block = conv_block
        net_identity_block = identity_block
    else:
        net_conv_block = conv_block_basic
        net_identity_block = identity_block_basic
    
    if bottleneck:
        x = net_conv_block(x, 64, stage=2, block='a', strides=(1, 1))
    else:
        x = net_identity_block(x, 64, stage=2, block='a')
    for i in range(num_block[0] - 1):
        block_id = chr(ascii_a + i + 1)
        x = net_identity_block(x, 64, stage=2, block=block_id,
                               weight_decay=weight_decay, use_bias=use_bias)

    x = net_conv_block(x, 128, stage=3, block='a',
                       weight_decay=weight_decay, use_bias=use_bias)
    for i in range(num_block[1] - 1):
        block_id = chr(ascii_a + i + 1)
        x = net_identity_block(x, 128, stage=3, block=block_id,
                               weight_decay=weight_decay, use_bias=use_bias)

    x = net_conv_block(x, 256, stage=4, block='a',
                       weight_decay=weight_decay, use_bias=use_bias)
    for i in range(num_block[2] - 1):
        block_id = chr(ascii_a + i + 1)
        if num_block[2] > 10:
            block_id = 'b' + str(i + 1)
        x = net_identity_block(x, 256, stage=4, block=block_id,
                               weight_decay=weight_decay, use_bias=use_bias)

    x = net_conv_block(x, 512, stage=5, block='a',
                       weight_decay=weight_decay, use_bias=use_bias)
    for i in range(num_block[3] - 1):
        block_id = chr(ascii_a + i + 1)
        x = net_identity_block(x, 512, stage=5, block=block_id,
                               weight_decay=weight_decay, use_bias=use_bias)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # x = layers.Dense(classes, activation='softmax', name='fc')(x)
    x = layers.Dense(classes, name='fc')(x)
    x = Activation('softmax')(x)
    return Model(main_input, x)



def resnet18():
    return resnet([2, 2, 2, 2], bottleneck=False)


def resnet34():
    return resnet([3, 4, 6, 3], bottleneck=False)


def resnet50():
    return resnet([3, 4, 6, 3])


def resnet101():
    return resnet([3, 4, 23, 3])


def resnet152():
    return resnet([3, 8, 36, 3])


if __name__ == "__main__":
    model = resnet34()
    model.summary()
