from datetime import time

import tensorboard as tensorboard
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Permute, Conv1D, MaxPooling1D
from keras.constraints import UnitNorm, MaxNorm
from keras import Sequential, optimizers, regularizers


class TextCNN(object):
    """
    A CNN for text salience scoring.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, train_data, train_labels, num_filters, kernel_size):

        print(train_data.shape)
        print(train_labels.shape)
        # Adding Keras layers to fit the model
        print("adding layers to fit model")
        model = Sequential()
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, strides=1, padding='valid',
                         data_format='channels_last', dilation_rate=1, activation='relu',
                         use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         batch_size=None,
                         input_shape=(train_data.shape[1], train_data.shape[2])))
        print("input shape", model.input_shape)
        model.add(MaxPooling1D(pool_size=train_data.shape[1] - (kernel_size - 1), strides=1, padding='valid',
                               data_format='channels_last'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', use_bias=True,
                        kernel_regularizer=regularizers.l2(0.01)))

        print("output shape", model.output_shape)
        ada = optimizers.Adadelta(lr=1)
        model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['mae'])
        print(model.summary())
        # sess.graph contains the graph definition; that enables the Graph Visualizer.
        # tb = TensorBoard(log_dir="logs/{}".format(time()))
        tb = TensorBoard(log_dir='./logs'.format(time()), histogram_freq=0, write_graph=True, write_images=False)
        print("fitting model to train")
        model.fit(train_data, train_labels, epochs=10, batch_size=None, validation_split=0.2, callbacks=[tb])

        tf.keras.backend.get_session().run(tf.global_variables_initializer())

        tf.keras.models.save_model(
            model,
            "trained_model",
            overwrite=True,
            include_optimizer=True
        )

    @staticmethod
    def eval(test_data):

        model = tf.keras.models.load_model("trained_model")
        results = model.predict(test_data, batch_size=None, verbose=1)
        return results
