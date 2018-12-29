from datetime import time

import tensorboard as tensorboard
import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import TensorBoard

class TextCNN(object):
    """
    A CNN for text salience scoring.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, train, test, sequence_length, num_filters, kernel_size):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Adding Keras layers to fit the model
        print("adding layers to fit model")
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(num_filters, kernel_size, strides=1, padding='valid',
                                      data_format='channels_last', dilation_rate=1, activation=tf.nn.sigmoid,
                                      use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                      kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                      kernel_constraint=None, bias_constraint=None))
        model.add(keras.layers.MaxPooling1D(pool_size=sequence_length - kernel_size + 1, strides=None, padding='valid',
                                            data_format='channels_last'))
        model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))
        #model.add(keras.activations.softmax(x=self.input_x, axis=-1))
        model.compile(optimizer='Adadelta', loss='mse')
        print("fitting model to train")
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        model.fit(train[0], train[1], epochs=10, batch_size=32, callbacks=[tensorboard])

        print("OK crystals")
        model.summary()
