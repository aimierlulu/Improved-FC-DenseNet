###############################################
# fully convolution DenseNet model
###############################################
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers.core import Dropout, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import tensorflow as tf###
from keras.layers import multiply
import matplotlib.pyplot as plt###
from keras.layers import Lambda###
import keras.backend as K

def DenseNetFCN(input_shape, nb_dense_block=3, growth_rate=16, nb_layers_per_block=4,
                reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, init_conv_filters=48,
                classes=1, upsampling_type='deconv'):

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv".')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. Argument '
                         'value was %d.' % (nb_layers_per_block))

    # Determine proper input shape
    min_size = 2 ** nb_dense_block

    if input_shape is not None:
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
                (input_shape[1] is not None and input_shape[1] < min_size)):
            raise ValueError('Input size must be at least ' +
                             str(min_size) + 'x' + str(min_size) +
                             ', got input_shape=' + str(input_shape))
    else:
        input_shape = (None, None, classes)

    img_input = Input(shape=input_shape)

    x = __create_fcn_dense_net(classes, img_input, nb_dense_block,
                               growth_rate, reduction, dropout_rate, weight_decay,
                               nb_layers_per_block, upsampling_type,
                               init_conv_filters, input_shape)
    inputs = img_input
    model = Model(inputs=inputs, outputs=x, name='fcn-densenet')
    return model

skip_down_list = []
def __create_fcn_dense_net(nb_classes, img_input, nb_dense_block=5, growth_rate=12,
                           reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                           nb_layers_per_block=4, upsampling_type='upsampling',
                           init_conv_filters=48, input_shape=None):
    concat_axis = -1
    rows, cols, _ = input_shape
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'
    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list
        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'

        # Add the deconv layers
        bottleneck_nb_layers = nb_layers[-1]
        # reverse to get the numbe of layers in DenseBlock in deconv path
        rev_layers = nb_layers[::-1]
        nb_layers.extend(rev_layers[1:])

    else:
        bottleneck_nb_layers = nb_layers_per_block
        nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

    # compute compression factor
    compression = 1.0 - reduction
    x1 = Conv2D(init_conv_filters, (3, 3), kernel_initializer='he_normal', padding='same', name='initial_conv2D1',
               kernel_regularizer=l2(weight_decay))(img_input)
    x2 = Conv2D(init_conv_filters, (5, 5), kernel_initializer='he_normal', padding='same', name='initial_conv2D2',
               kernel_regularizer=l2(weight_decay))(img_input)###
    x = concatenate([x1, x2], axis=concat_axis)
    x = Conv2D(init_conv_filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)###
    x = Conv2D(init_conv_filters, (3, 3), kernel_initializer='he_normal', padding='same', name='initial_conv2D',
               kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    nb_filter = init_conv_filters
    skip_list = []
    
    skip_down_list.append(x)
    # Add dense blocks and transition down block
    for block_idx in range(nb_dense_block):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)
        # Skip connection
        skip_list.append(x)
        # add transition_block
        x = __transition_block(x, nb_filter, dropout_rate=dropout_rate,
                               compression=compression, weight_decay=weight_decay)
        skip_down_list.append(x)###
        nb_filter = int(nb_filter * compression)
    _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers, nb_filter, growth_rate,
                                              dropout_rate=dropout_rate, weight_decay=weight_decay,
                                              return_concat_list=True)
    skip_list = skip_list[::-1]
   
    # Add dense blocks and transition up block  
    for block_idx in range(nb_dense_block):
        n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

        l = concatenate(concat_list[1:], axis=concat_axis)

        t = __transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay)
        t1= __transition_up_block1(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay)
       

        a,b,c,d = K.int_shape(t)
        t = concatenate([t, t1], axis=concat_axis)
        t = Conv2D(d, (3, 3), kernel_initializer='he_normal', padding='same')(t)###
        x = concatenate([t, skip_list[block_idx]], axis=concat_axis)   

        x, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1],
                                                  nb_filter=growth_rate,
                                                  growth_rate=growth_rate, dropout_rate=dropout_rate,
                                                  weight_decay=weight_decay, return_concat_list=True,
                                                  grow_nb_filters=False)

    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same', use_bias=False)(x)
    return x
def __conv_block(input, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):

    concat_axis = -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4

        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    #x1 = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    #x2 = Conv2D(nb_filter, (5, 5), kernel_initializer='he_normal', padding='same')(x)###
    #x = concatenate([x1, x2], axis=concat_axis)###
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same')(x)###
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    concat_axis = -1

    x_list = [x]
    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if nb_layers == 5:
       t1 = skip_down_list[0]
       t1 = AveragePooling2D((2, 2), strides=(2, 2))(t1)
       x = concatenate([x,t1],axis=concat_axis) 
       nb_filter +=48
    else:
       if nb_layers == 7:
          t1 = skip_down_list[0]
          t1 = AveragePooling2D((2, 2), strides=(2, 2))(t1)
          t2 = skip_down_list[1]
          t2 = AveragePooling2D((2, 2), strides=(2, 2))(t2)
          t3 = AveragePooling2D((2, 2), strides=(2, 2))(t1)
          x = concatenate([x,t2],axis=concat_axis) 
          x = concatenate([x,t3],axis=concat_axis)
          nb_filter = nb_filter+48+112;
    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(input, nb_filter, dropout_rate, compression=1.0, weight_decay=1e-4):
    '''
        transition_block = BN + Relu + Conv2D + AVGPooling
    '''
    concat_axis = -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same',
               kernel_regularizer=l2(weight_decay))(x)
   
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x

def __transition_up_block(input, nb_filters, type='deconv', weight_decay=1E-4):

    concat_axis = -1
    if type == 'upsampling':
        x = UpSampling2D(size=(2, 2))(input)
    else:
        x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
        
    return x

def __transition_up_block1(input, nb_filters, type='deconv', weight_decay=1E-4):

    concat_axis = -1
    if type == 'upsampling':
        x = UpSampling2D(size=(2, 2))(input)
    else:
        x = Conv2DTranspose(nb_filters, (5, 5), activation='relu', padding='same', strides=(2, 2),
                            kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    return x

if __name__ == '__main__':
    model = DenseNetFCN((128, 128, 3), growth_rate=16, nb_layers_per_block=[4, 5, 7, 10, 12, 15],
                        upsampling_type='deconv', dropout_rate=0.2)
    model.summary()

    plot_model(model, './cur_model.png')
