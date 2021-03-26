###############################################
# for easy model --U-net --fcn-vgg16
###############################################
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers import BatchNormalization
from keras.layers.noise import GaussianNoise
import keras

np.random.seed(4)


def unet(img_rows, img_cols, loss, optimizer, metrics, fc_size=8192, channels=3):
    filter_size = 5
    filter_size_2 = 11
    dropout_a = 0.5
    dropout_b = 0.5
    dropout_c = 0.5
    gaussian_noise_std = 0.025

    inputs = Input((img_rows, img_cols, channels))
    input_with_noise = GaussianNoise(gaussian_noise_std)(inputs)

    conv1 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(input_with_noise)
    conv1 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    pool1 = GaussianNoise(gaussian_noise_std)(pool1)

    conv2 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    pool2 = GaussianNoise(gaussian_noise_std)(pool2) 

    conv3 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(dropout_a)(pool3)

    fc = Flatten()(pool3)
    fc = Dense(fc_size, activation='relu')(fc)
    fc = Dropout(dropout_b)(fc)

    n = img_rows // 2 // 2 // 2
    fc = Dense(128 * n * n, activation='relu')(fc)
    fc = GaussianNoise(gaussian_noise_std)(fc)
    fc = Reshape((n, n, 128))(fc)

    up1 = concatenate([UpSampling2D(size=(2, 2))(fc), conv3], axis=3)
    up1 = Dropout(dropout_c)(up1)

    conv4 = Conv2D(128, kernel_size=(filter_size_2, filter_size_2), activation='relu', padding='same')(up1)
    conv4 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv2], axis=3)
    up2 = Dropout(dropout_c)(up2)

    conv5 = Conv2D(64, kernel_size=(filter_size_2, filter_size_2), activation='relu', padding='same')(up2)
    conv5 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv5)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv1], axis=3)
    up3 = Dropout(dropout_c)(up3)

    conv6 = Conv2D(32, kernel_size=(filter_size_2, filter_size_2), activation='relu', padding='same')(up3)
    conv6 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', padding='same')(conv6)

    conv7 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    return model


def vgg16(img_rows, img_cols, pretrained, freeze_pretrained, loss, optimizer, metrics, channels=3):
    if pretrained:
        assert channels == 3, 'if pretrained, channels must be 3'
        base_model = keras.applications.VGG16(weights='imagenet', include_top=False, pooling=None,
                                              input_shape=(img_rows, img_cols, channels))

        conv1 = base_model.get_layer('block1_conv2').output
        conv2 = base_model.get_layer('block2_conv2').output
        conv3 = base_model.get_layer('block3_conv3').output
        conv4 = base_model.get_layer('block4_conv3').output
        conv5 = base_model.get_layer('block5_conv3').output

        if freeze_pretrained:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = True
        model = base_model
    else:
        inputs = Input(shape=(img_rows, img_cols, channels))

        conv1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
        conv1 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2')(conv1)
        pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

        conv2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1')(pool1)
        conv2 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2')(conv2)
        pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

        conv3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1')(pool2)
        conv3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2')(conv3)
        conv3 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3')(conv3)
        pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

        conv4 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1')(pool3)
        conv4 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2')(conv4)
        conv4 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3')(conv4)
        pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

        conv5 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1')(pool4)
        conv5 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_2')(conv5)
        conv5 = Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_3')(conv5)

        model = Model(inputs=inputs, outputs=conv5)

    dropout_val = 0.5
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    up6 = Dropout(dropout_val)(up6)

    conv6 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(up6)
    conv6 = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    up7 = Dropout(dropout_val)(up7)

    conv7 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(up7)
    conv7 = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    up8 = Dropout(dropout_val)(up8)

    conv8 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(up8)
    conv8 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    up9 = Dropout(dropout_val)(up9)

    conv9 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(up9)
    conv9 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(conv9)

    conv10 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=model.input, outputs=conv10)

    return model


def unet2(img_rows, img_cols, loss, optimizer, metrics, fc_size=0, channels=3):
    filter_size = 5
    filter_size_2 = 11
    dropout_a = 0.5
    dropout_b = 0.5
    dropout_c = 0.5
    gaussian_noise_std = 0.025

    inputs = Input((channels, img_rows, img_cols))
    input_with_noise = GaussianNoise(gaussian_noise_std)(inputs)

    conv1 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(input_with_noise)
    conv1 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv1)
    conv1 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    pool1 = GaussianNoise(gaussian_noise_std)(pool1)

    conv2 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv2)
    conv2 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    pool2 = GaussianNoise(gaussian_noise_std)(pool2)

    conv3 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv3)
    conv3 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(dropout_a)(pool3)
    if fc_size > 0:
        fc = Flatten()(pool3)
        fc = Dense(fc_size)(fc)
        fc = BatchNormalization()(fc)
        fc = Activation('relu')(fc)
        fc = Dropout(dropout_b)(fc)

        n = img_rows // 2 // 2 // 2
        fc = Dense(img_rows * n * n)(fc)
        fc = BatchNormalization()(fc)
        fc = Activation('relu')(fc)
        fc = GaussianNoise(gaussian_noise_std)(fc)
        fc = Reshape((128, n, n))(fc)
    else:
        fc = Conv2D(256, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(pool3)
        fc = BatchNormalization()(fc)
        fc = Dropout(dropout_b)(fc)

    up1 = concatenate([UpSampling2D(size=(2, 2))(fc), conv3], axis=3)
    up1 = BatchNormalization()(up1)
    up1 = Dropout(dropout_c)(up1)

    conv4 = Conv2D(128, kernel_size=(filter_size_2, filter_size_2), activation='relu', border_mode='same')(up1)
    conv4 = Conv2D(128, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv4)
    conv4 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv4)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv2], axis=3)
    up2 = BatchNormalization()(up2)
    up2 = Dropout(dropout_c)(up2)

    conv5 = Conv2D(64, kernel_size=(filter_size_2, filter_size_2), activation='relu', border_mode='same')(up2)
    conv5 = Conv2D(64, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv5)
    conv5 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv5)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv1], axis=3)
    up3 = BatchNormalization()(up3)
    up3 = Dropout(dropout_c)(up3)

    conv6 = Conv2D(32, kernel_size=(filter_size_2, filter_size_2), activation='relu', border_mode='same')(up3)
    conv6 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv6)
    conv6 = Conv2D(32, kernel_size=(filter_size, filter_size), activation='relu', border_mode='same')(conv6)

    conv7 = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    return model
