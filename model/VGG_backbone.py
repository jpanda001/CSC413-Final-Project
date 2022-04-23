import tensorflow as tf
import tensorflow.keras as K
import model.layers as L

def VGGConvLayers(threeLayers=False, input_features, output_channels, strides=(1,1), \
                    bias=True, batchNormalization=False, pooling=True):
    """ VGG: 
            2 conv layers 
            1 pool layer 
    """
    convOut = L.convolution2D(input_features, output_channels, 3, strides, activation=True, \
                            biases=bias, batchNormalization=batchNormalization, \
                            regularization=True)
    convOut = L.convolution2D(convOut, output_channels, 3, strides, activation=True, \
                            biases=bias, batchNormalization=batchNormalization, \
                            regularization=True)
    if threeLayers:
        convOut = L.convolution2D(convOut, output_channels, 3, strides, activation=True, \
                            biases=bias, batchNormalization=batchNormalization, \
                            regularization=True)
    if pooling:
        convOut = L.maxPooling2D(convOut)

    return convOut

def VGG(input_features):
    """ VGG main (customized) """
    
    ##### VGG16#####
    convOut = VGGConvLayers(False, input_features, int(conv.shape[-1]), pooling=False)
    convOut = VGGConvLayers(False, convOut, int(conv.shape[-1]), pooling=True)
    convOut = VGGConvLayers(True, convOut, int(conv.shape[-1]*2), pooling=True)
    convOut = VGGConvLayers(True, convOut, int(conv.shape[-1]*2), pooling=True)
    convOut = VGGConvLayers(True, convOut, int(conv.shape[-1]*2), pooling=True)

    return convOut
