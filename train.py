import os
import pdb
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, History
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data_generate
from metrics import tversky_loss, focal_loss, dice_loss, jacc_loss, jacc_coef, dice_coef, dice_coefficient_loss, precision
import models
import argparse
import glob
import re
import keras

import tensorflow as tf
# old_v = tf.compat.v1.logging.get_verbosity()
old_v = tf.logging.get_verbosity()

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

#from keras.backend.tensorflow_backend import set_session
import fc_densenet_three_improvements #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#import fc_densenet
from keras.utils.vis_utils import plot_model
seed = 1
train_dir = "./train/"
train_mask_dir = "./train_mask/"
validation_while_training = "./validation/"
validation_mask_while_training = "./validation_mask/"

model_path = "./trainedModel/"

loss_options = {'BCE': 'binary_crossentropy', 'dice': dice_loss, 'jacc': jacc_loss, 'MSE': 'mean_squared_error', 'focal': 'focal_loss', 'tversky' :'tversky_loss', 'dice_coefficient': 'dice_coefficient_loss'}
optimizer_options = {'adam': Adam(lr=0.00006),  
                     'sgd': SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)}

parser = argparse.ArgumentParser()

parser.add_argument("--train_from_scratch", type=bool, default=True)

parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)

parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--epoch", type=int, default=100)###############

parser.add_argument("--loss_param", type=str, default="tversky", choices=['jacc', 'dice', 'BCE', 'MSE', 'focal', 'tversky', 'dice_coefficient'])#####################

parser.add_argument("--training_model", type=str, default="fcDenseNet", choices=['unet', 'unet2', 'vgg', 'fcDenseNet'])
parser.add_argument("--optimizer_param", type=str, default="sgd", choices=['adam', 'sgd'])

# 标准化
parser.add_argument("--remove_mean_imagenet", type=bool, default=True)

parser.add_argument("--rescale_mask", type=bool, default=True)
parser.add_argument("--use_hsv", type=bool, default=False)

metrics = [dice_coef, jacc_coef]

# only for vgg16
parser.add_argument("--pretrained", type=bool, default=True)
parser.add_argument("--freeze_pretrained", type=bool, default=False)

# only for Unet
parser.add_argument("--fc_size", type=int, default=8192)

# only for FC_DenseNet
parser.add_argument("--growth_rate", type=int, default=16)
parser.add_argument("--upsampling_type", type=str, default="deconv", choices=['upsampling', 'deconv'])

#nb_layers_per_block = [4, 5, 7, 10, 12, 15]
nb_layers_per_block = [4, 5, 7, 10]
def my_generator(train_generator, train_mask_generator):
    while True:
        train_gen = next(train_generator)
        train_mask = next(train_mask_generator)

        if False:  # use True to show images
            mask_true_show = np.where(train_mask >= 0.5, 1, 0)
            mask_true_show = mask_true_show * 255
            mask_true_show = mask_true_show.astype(np.uint8)
            for i in range(train_gen.shape[0]):
                mask = mask_true_show[i].reshape((FLAGS.width, FLAGS.height))
                img = train_gen[i]
                img = img[0:3]
                img = img.astype(np.float16)###
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(img)
                ax[0].axis("off")
                ax[1].imshow(mask, cmap='Greys_r')
                ax[1].axis("off")
                plt.show()
        yield (train_gen, train_mask)


def load_data():
    print("Prepare dataset")

    train_list = os.listdir(train_dir)
    val_while_train_list = os.listdir(validation_while_training)
    print("Start loading images")

    train, train_mask = data_generate.load_images(train_list, FLAGS.height, FLAGS.width,
                                                  train_dir, train_mask_dir,
                                                  remove_mean_imagenet=FLAGS.remove_mean_imagenet,
                                                  rescale_mask=FLAGS.rescale_mask, use_hsv=FLAGS.use_hsv
                                                  )
    val_train, val_train_mask = data_generate.load_images(val_while_train_list, FLAGS.height, FLAGS.width,
                                                          validation_while_training, validation_mask_while_training,
                                                          remove_mean_imagenet=FLAGS.remove_mean_imagenet,
                                                          rescale_mask=FLAGS.rescale_mask, use_hsv=FLAGS.use_hsv
                                                          )

    print("Load images done")
    return train, train_mask, val_train, val_train_mask


def training(FLAGS, n_channels):
    train, train_mask, val_train, val_train_mask = load_data()#文件名列表

    print("Trainging data: " + str(len(train)))
    print("Validation_while_train: " + str(len(val_train)))
    print("Using batch size = {}".format(FLAGS.batch_size))

    #train_data_gen_args = dict(featurewise_center=False,
    #                           featurewise_std_normalization=False,
    #                           rotation_range=90,
    #                           width_shift_range=0.1,
    #                           height_shift_range=0.1,
    #                           horizontal_flip=True,
    #                           vertical_flip=True,
    #                           zoom_range=0.2,
    #                           fill_mode='reflect',
    #                           )
   
    train_data_gen_args = dict(featurewise_center=False,
                               featurewise_std_normalization=False,
                               rotation_range=0,
                               width_shift_range=0,
                               height_shift_range=0,
                               horizontal_flip=False,
                               vertical_flip=False,
                               zoom_range=0,
                               fill_mode='reflect',
                               )
    train_data_gen_mask_args = dict(train_data_gen_args.items())
    train_data_gen_mask_args['fill_mode'] = 'nearest'

    val_data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             fill_mode='reflect',
                             )

    val_data_gen_mask_args = dict(val_data_gen_args.items())
    val_data_gen_mask_args['fill_mode'] = 'nearest'
    #############################################

    print("Create Data Generator")

    train_datagen = ImageDataGenerator(**train_data_gen_args)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_mask_args)

    validation_datagen = ImageDataGenerator(**val_data_gen_args)
    validation_mask_datagen = ImageDataGenerator(**val_data_gen_mask_args)
    train_generator = train_datagen.flow(train, batch_size=FLAGS.batch_size, seed=seed)
   
    train_mask_generator = train_mask_datagen.flow(train_mask, batch_size=FLAGS.batch_size, seed=seed)

    validation_generator = validation_datagen.flow(val_train, batch_size=FLAGS.batch_size, seed=seed)
    validation_mask_generator = validation_mask_datagen.flow(val_train_mask, batch_size=FLAGS.batch_size, seed=seed)

    train_generator_flow = my_generator(train_generator, train_mask_generator)
    
    
    validation_generator_flow = my_generator(validation_generator, validation_mask_generator)

    loss = loss_options[FLAGS.loss_param]
    optimizer = optimizer_options[FLAGS.optimizer_param]

    ########## callbacks ##########
    checkpoint_file = "weights.{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5"
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_path, checkpoint_file),
                                       monitor='val_loss',
                                       save_best_only=False,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1,
                                       verbose=2##########
                                       )
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='min')###########
    tensor_board = TensorBoard(log_dir=model_path, write_images=True, write_graph=True)
    history = History()
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=10, verbose=2, mode='min',
                                                  min_lr=1e-6)#########
    ###############################    

    if FLAGS.train_from_scratch:
        print('Create model---' + FLAGS.training_model)

        if FLAGS.training_model == 'unet':
            model = models.unet(FLAGS.height, FLAGS.width, loss=loss, optimizer=optimizer,
                                metrics=metrics, fc_size=FLAGS.fc_size, channels=n_channels)
        elif FLAGS.training_model == 'unet2':
            model = models.unet2(FLAGS.height, FLAGS.width, loss=loss, optimizer=optimizer,
                                 metrics=metrics, fc_size=FLAGS.fc_size, channels=n_channels)
        elif FLAGS.training_model == 'vgg':
            model = models.vgg16(FLAGS.height, FLAGS.width,
                                 pretrained=FLAGS.pretrained,
                                 freeze_pretrained=FLAGS.freeze_pretrained,
                                 loss=loss, optimizer=optimizer, metrics=metrics, channels=n_channels)
        elif FLAGS.training_model == 'fcDenseNet':
            model = fc_densenet_three_improvements.DenseNetFCN((FLAGS.height, FLAGS.width, n_channels),          #!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                            growth_rate=FLAGS.growth_rate,
                                            upsampling_type=FLAGS.upsampling_type,
                                            nb_layers_per_block=nb_layers_per_block,
                                            dropout_rate=0.2
                                            )
        else:
            model = None
            print("Please choose right model!")
            return
        # plot_model(model, to_file='./{}.png'.format(FLAGS.training_model), show_shapes=True)
        model.compile(loss=[tversky_loss], optimizer=optimizer, metrics=metrics)
        #model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.summary()
        #############################

        hist=model.fit_generator(
            train_generator_flow,
            steps_per_epoch=len(train) // FLAGS.batch_size,
            epochs=FLAGS.epoch,
            verbose=2,
            validation_data=validation_generator_flow,
            validation_steps=len(val_train) // FLAGS.batch_size,
            shuffle=True,
            callbacks=[model_checkpoint,tensor_board, history]
        )
        with open('log.txt','w') as f:
            f.write(str(hist.history))
        plt.plot(hist.history['loss'])

        plt.plot(hist.history['val_loss'])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train","test"],loc="upper left")
        plt.savefig('loss.jpg')
        plt.show()

        ################
    else:
        print("Load model...")
        model_files = glob.glob(model_path + "weights*.hdf5")
        assert len(model_files) != 0, 'There is no model to load!'

        pre_epoch = 1
        model_file = ""
        for file in model_files:
            epoch = re.findall("\d+", file.split("-")[0])
            epoch = int(epoch[0])
            if epoch >= pre_epoch:
                pre_epoch = epoch
                model_file = file

        print("Continue training from", pre_epoch, "epoch...")

        model = keras.models.load_model(model_file, custom_objects={'dice_loss': dice_loss,
                                                                    'dice_coef': dice_coef,
                                                                    'jacc_loss': jacc_loss,
                                                                    'jacc_coef': jacc_coef,
                                                                    'tversky_loss': tversky_loss,
                                                                    'focal_loss': focal_loss,
                                                                    'dice_coefficient_loss': dice_coefficient_loss})

        model.compile(loss=[tversky_loss], optimizer=optimizer, metrics=metrics)
        hist=model.fit_generator(
            train_generator_flow,
            steps_per_epoch=len(train) // FLAGS.batch_size,
            epochs=pre_epoch + FLAGS.epoch,
            verbose=2, 
            validation_data=validation_generator_flow,
            validation_steps=len(val_train) // FLAGS.batch_size,
            shuffle=True,
            callbacks=[model_checkpoint, tensor_board, history],
            initial_epoch=pre_epoch,
        )
        with open('log.txt','w') as f:
            f.write(str(hist.history))
        plt.plot(hist.history['loss'])

        plt.plot(hist.history['val_loss'])

        plt.title("model loss")

        plt.ylabel("loss")

        plt.xlabel("epoch")

        plt.legend(["train","validation"],loc="upper left")
        plt.savefig('loss.jpg')
        plt.show()

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.use_hsv:
        n_channels = 6
        print("Using HSV")
    else:
        n_channels = 3###########################

    training(FLAGS, n_channels)
    print(n_channels)
