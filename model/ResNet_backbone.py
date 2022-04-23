from cv2 import repeat
import tensorflow as tf
import tensorflow.keras as K
import model.layers as L


def residualBlock(input_features, channel_expansion, strides=(1,1), use_bias=False):
    """ Basic block of resnet3d """
    input_channel = x.shape[-1]
    convOut = L.convolution2D(input, input_channel, 3, strides, \
                            activation=True, biases=use_bias, batchNormalization=True)
    convOut = L.convolution2D(convOut, int(input_channel*channel_expansion), 3, \
                            (1,1), activation=True, biases=use_bias, batchNormalization=True)
    convOut = L.convolution2D(convOut, convOut.shape[-1], 1, (1,1), \
                            activation=True, biases=use_bias, batchNormalization=True)

    ### combine all together to generate residual feature maps ###
    convOut += input_features
    return convOut

def mainBlock(input_features, repeats, layerStructure):
    """ Iterative block """
    """ 
    Residual blocks; 
    Upsample block; 
    """
    ##### Start repeated block #####
    all_strides, all_expansions = layerStructure[0], layerStructure[1]
    convOut - input_features

    for i in range(repeats):
        strides = (all_strides[i], all_strides[i])
        expansion = all_expansions[i]
        convOut = residualBlock(convOut, expansion, strides, use_bias=True) 
    
    convOut = L.maxPooling2D(convOut)
    
    return convOut

def ResNet(input):
    """ Build backbine ResNet3D """
    ##### Parameters #####
    # repeating and upsampling of blocks in the network
    networkStructure = [(2, False), (4, False), (8, True), (16, True)]
    convOut = input
    ##### repeating and combining residual blocks ######
    for i in range(len(networkStructure)):
        repeats = networkStructure[i][0]
        strides = [1, 1] * int(networkStructure[i][0]/2) # [1, 1, 1, 1] 
        expansions = strides
        
        if networkStructure[i][1] == True: 
            expansions[-1] *= 2

        convOut = mainBlock(convOut, repeats, [strides, expansions])

    return convOut
