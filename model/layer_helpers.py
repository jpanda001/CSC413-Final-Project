import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import os

def convolution2D(input_features, output_channels, kernel, strides, activation=True, \
                    biases=True, batchNormalization=True, regularization=False):
    """ 2D convolutional layer"""

    # reducing overfitting with regularization
    if regularization:
        kernel_regularizer = K.regularizers.L2(1e-2)
        if use_bias:
            bias_regularizer = K.regularizers.L2(1e-2)
        else:
            bias_regularizer = None
    else:
        kernel_regularizer = None
        bias_regularizer = None

    conv_output = K.layers.Conv2D(filters=output_channels, kernel_size=kernel_size, strides=strides, \
        padding="same", activation=None, use_bias=use_bias, kernel_initializer='glorot_uniform', 
        bias_initializer='zeros', kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer,)(input_features)

    if batchNormalization: 
        conv_output = BatchNormalization()(conv_output)
    
    if activation:
        conv_output = tf.nn.relu(conv_output)

    return conv_output


def maxPooling2D(input_features, strides=(2,2), pool_size=(2,2)):
    """ Max pooling layer"""

    pool_output = tf.keras.layers.MaxPool2D(pool_size=pool_size, \
        strides=strides, padding="same")(input_features)
    
    return pool_output


class BatchNormalization(K.layers.BatchNormalization):
    """
    Implementing Batch Normalization measure used in YOLO
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)
