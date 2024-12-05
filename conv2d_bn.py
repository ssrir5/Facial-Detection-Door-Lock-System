
'''
*Definition for 2 dimension convolution layer with batch normalization*
'''

from modules import *


def conv2d_bn(x,
              filters, # number of output channels
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):

    ## DO NOT TOUCH
    bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
    ##############################################################

## Apply a Conv 2D Keras Layer 
    x = Conv2D(filters, kernel_size, strides, padding, use_bias=use_bias, name=name)(x)

    if not use_bias:

        bn_name = generate_layer_name('BatchNorm', prefix=name)
        ## Apply a Batch Normalization Keras Layer 
        x = BatchNormalization(bn_axis, 0.995, 0.001, scale=False, name=bn_name)(x)



    if activation is not None:
        ac_name = generate_layer_name('Activation', prefix=name)
        ## Apply an Activation Keras Layer
        x = Activation(activation, name=ac_name)(x)



    ###############################################################
    return x 
