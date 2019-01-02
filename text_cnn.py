from datetime import time

# import tensorboard as tensorboard
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.constraints import UnitNorm
from keras import Sequential
from sklearn.datasets import make_blobs


class TextCNN(object):
    """
    A CNN for text salience scoring.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, train_data, train_labels, test_data, test_labels, num_filters, kernel_size):

        #X, y = make_blobs(n_samples=1000, centers=2, n_features=300, random_state=1)
        #X_test, y_test = make_blobs(n_samples=1000, centers=2, n_features=300, random_state=1)
        #X.shape = (X.shape[0], X.shape[1], 1)
        #y.shape = (y.shape[0], 1)
        #X_test.shape = (X_test.shape[0], X_test.shape[1], 1)
        #y_test.shape = (y_test.shape[0], 1)
        #print(X.shape)
        #print(y.shape)

        print(train_data.shape)
        print(train_labels.shape)
        print(test_data.shape)
        print(test_labels.shape)


        # Adding Keras layers to fit the model
        print("adding layers to fit model")
        model = Sequential()
        model.add(Conv2D(filters=num_filters, kernel_size=(kernel_size, 300), strides=1, padding='valid',
                         data_format='channels_last', dilation_rate=1, activation=tf.nn.sigmoid,
                         use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                         kernel_constraint=None, bias_constraint=None, input_shape=(train_data.shape[1], train_data.shape[2], 1)))
        print("input shape", model.input_shape)
        model.add(MaxPooling2D(pool_size=(train_data.shape[1] - kernel_size + 1, 1), strides=1, padding='valid',
                               data_format='channels_last'))
        model.add(Flatten())

        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', use_bias=True, kernel_constraint=UnitNorm(axis=0)))

        print("output shape", model.output_shape)
        # model.add(keras.activations.softmax(x=self.input_x, axis=-1))
        model.compile(optimizer='Adadelta', loss='mse')
        print(model.summary())
        # sess.graph contains the graph definition; that enables the Graph Visualizer.
        # file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph)

        # tb = TensorBoard(log_dir="logs/{}".format(time()))
        tb = TensorBoard(log_dir='./logs'.format(time()), histogram_freq=0, write_graph=True, write_images=False)
        print("fitting model to train")
        model.fit(train_data, train_labels, epochs=10, batch_size=None, callbacks=[tb])
        print(model.evaluate(test_data, test_labels, verbose=1))
