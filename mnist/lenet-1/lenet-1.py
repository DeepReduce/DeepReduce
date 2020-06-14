import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,Activation
from keras.callbacks import LearningRateScheduler, TensorBoard
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


KTF.set_session(session)

batch_size = 256
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def build_model():
    model = Sequential()

    model.add(Conv2D(4, (5, 5), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(12, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    # model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    times = int(sys.argv[1])
    # load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 处理 x
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # 处理 y
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    # build network
    model = build_model()
    print(model.summary())

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    # save model
    path = 'Cov/model/'
    file_name = 'lenet-1_' + str(times) + '.h5'
    model.save(path + file_name)

