###############################################
# file to load data
###############################################
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from random import shuffle
import argparse
import sys
import tensorflow as tf

np.random.seed(4)
mean_imagenet = [123.68, 103.939, 116.779]  # rgb
#mean_imagenet = [0]

parser = argparse.ArgumentParser()

parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--validation_percent", type=float, default=0.2)
parser.add_argument("--regenerate_data", type=bool, default=True)

training_folder = "./datasets9/train/"
training_mask_folder = "./datasets9/train_mask/"

test_folder = "./datasets9/test/"
test_mask_folder = "./datasets9/test_mask/"


def get_mask(image_name, mask_folder, rescale_mask=True):
    img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg", "_segmentation.png")),
                          cv2.IMREAD_GRAYSCALE)
    if img_mask is None:
        img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg", ".png")),
                              cv2.IMREAD_GRAYSCALE)
    _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
    if rescale_mask:
        img_mask = img_mask / 255.
    return img_mask



def get_color_image(image_name, image_folder, remove_mean_imagenet=True, use_hsv=False):
    #print(os.path.join(image_folder, image_name))
    img = cv2.imread(os.path.join(image_folder, image_name))
    #r,g,b = cv2.split(img)
    #print(r.shape)
    #r = r.reshape((512,512,1))
    if img is None:
       img = cv2.imread(os.path.join(image_folder, image_name.replace(".jpg", ".png")))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float16)###############改了
    #img = img.reshape((512, 512, 1))
    if remove_mean_imagenet:
        for channel in [0, 1, 2]:
          img[:, channel] -= mean_imagenet[channel]
    if use_hsv:
        img_all = np.zeros((img.shape[0], img.shape[1], 6))
        img_all[:, :, 0:3] = img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_all[:, :, 3:] = img_hsv
        img = img_all
        
    img = img.astype(np.float16)############
    return img

def load_images(images_list, height, width, image_folder, mask_folder,
                remove_mean_imagenet=True, rescale_mask=True, use_hsv=False):
    if use_hsv:
        n_chan = 6
    else:
        n_chan = 3#############
    img_array = np.zeros((len(images_list), height, width, n_chan), dtype=np.float16)################
    img_mask_array = np.zeros((len(images_list), height, width), dtype=np.uint8)###############
    for i, image_name in enumerate(images_list):
        sys.stdout.write('\r>> Loading image %d/%d' % (
            i + 1, len(images_list)))
        sys.stdout.flush()

        img = get_color_image(image_name, image_folder, remove_mean_imagenet=remove_mean_imagenet, use_hsv=use_hsv)
        #img_array[i] = img
        img_array[i] = img/255###########??????????????????????????????
        if mask_folder:
            img_mask = get_mask(image_name, mask_folder, rescale_mask)
            img_mask_array[i] = img_mask
    print("\n")

    if not mask_folder:
        return img_array
    else:
        img_mask_array = img_mask_array.reshape(img_mask_array.shape[0], img_mask_array.shape[1], img_mask_array.shape[2], 1)##############
        return img_array, img_mask_array.astype(np.uint8)

def list_from_folder(image_folder):
    image_list = []
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".png"):
            image_list.append(image_filename)
        elif image_filename.endswith("jpg"):
            image_list.append(image_filename)
    print("Found {} images.".format(len(image_list)))
    return image_list


def move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder,
                height=None, width=None):
    for k in range(len(images_list)):
        sys.stdout.write('\r>> Resizing image %d/%d' % (
            k + 1, len(images_list)))
        sys.stdout.flush()
        image_filename = images_list[k]
        image_name = os.path.basename(image_filename).split('.')[0]

        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        if not os.path.exists(output_mask_folder):
            os.makedirs(output_mask_folder)

        img = cv2.imread(os.path.join(input_image_folder, image_filename))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_image_folder, image_name + ".png"), img)

        img_mask = get_mask(image_filename, input_mask_folder, rescale_mask=False)
        img_mask = cv2.resize(img_mask, (width, height), interpolation=cv2.INTER_CUBIC)
        #_, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_mask_folder, image_name + ".png"), img_mask)
    print("\n   Finished")


def resize_images(images_list, input_image_folder, input_mask_folder,
                  output_image_folder, output_mask_folder, height, width):
    return move_images(images_list, input_image_folder, input_mask_folder,
                       output_image_folder, output_mask_folder, height, width)


def get_mask_full_sized(mask_pred, original_shape, output_folder=None, image_name=None):
    mask_pred = cv2.resize(mask_pred, (original_shape[1], original_shape[0]))  # resize to original mask size
    #_, mask_pred = cv2.threshold(mask_pred, 127, 255, cv2.THRESH_BINARY)#???
    if output_folder and image_name:
        cv2.imwrite(os.path.join(output_folder, image_name.split('.')[0] + "_segmentation_out.png"), mask_pred)
    return mask_pred


def show_images_full_sized(image_list, img_mask_pred_array, image_folder, mask_folder, index,
                           output_folder=None, plot=True):
    image_name = image_list[index]
    img = get_color_image(image_name, image_folder, remove_mean_imagenet=False).astype(np.float16)###
    mask_pred = get_mask_full_sized(img_mask_pred_array[index], img.shape,
                                    output_folder=output_folder, image_name=image_name)
    if mask_folder:
        mask_true = get_mask(image_name, mask_folder, rescale_mask=False)
        if plot:
            f, ax = plt.subplots(1, 3)
            ax[0].imshow(img)
            ax[0].axis("off")
            ax[1].imshow(mask_true, cmap='Greys_r')
            ax[1].axis("off")
            ax[2].imshow(mask_pred, cmap='Greys_r')
            ax[2].axis("off")
            plt.show()
        return img, mask_true, mask_pred
    else:
        if plot:
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(img)
            ax[0].axis("off")
            ax[1].imshow(mask_pred, cmap='Greys_r')
            ax[1].axis("off")
            plt.show()
        return img, mask_pred


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    base_folder = "./data_{}_{}".format(FLAGS.height, FLAGS.width)
    if os.path.exists(base_folder):
        print("Already finish data resize!")
    else:
        print("Start data resize...")
        train_dir = os.path.join(base_folder, "train_image")
        train_mask_dir = os.path.join(base_folder, "train_mask")

        test_dir = os.path.join(base_folder, "test_image")
        test_mask_dir = os.path.join(base_folder, "test_mask")

        train_list = os.listdir(training_folder)
        test_list = os.listdir(test_folder)

        resize_images(train_list,
                      input_image_folder=training_folder,
                      input_mask_folder=training_mask_folder,
                      output_image_folder=train_dir, output_mask_folder=train_mask_dir,
                      height=FLAGS.height, width=FLAGS.width
                      )
        resize_images(test_list,
                      input_image_folder=test_folder,
                      input_mask_folder=test_mask_folder,
                      output_image_folder=test_dir,
                      output_mask_folder=test_mask_dir,
                      height=FLAGS.height, width=FLAGS.width
                      )
        print("   Finish data resize")

    if FLAGS.regenerate_data:
        print("Generate the trainging data and validation data...")
        train_dir = os.path.join(base_folder, "train_image")
        train_mask_dir = os.path.join(base_folder, "train_mask")
        train_list = os.listdir(train_dir)
        shuffle(train_list)
        NUM_VALIDATION = int(FLAGS.validation_percent * len(train_list))
        validation = train_list[:NUM_VALIDATION]
        train_list = train_list[NUM_VALIDATION:]

        for i, file in enumerate(train_list):
            sys.stdout.write('\r>> Loading image %d/%d' % (
                i + 1, len(train_list)))
            sys.stdout.flush()

            src = os.path.join(train_dir, file)
            src_mask = os.path.join(train_mask_dir, file)

            folder = "./data/train/"
            mask_folder = "./data/train_mask/"

            if not os.path.exists(folder):
                os.makedirs(folder)
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            dst = os.path.join(folder, file)
            dst_mask = os.path.join(mask_folder, file)

            shutil.copyfile(src, dst)
            shutil.copyfile(src_mask, dst_mask)
        print("\n   Finish generate training data")

        for i, file in enumerate(validation):
            sys.stdout.write('\r>> Loading image %d/%d' % (
                i + 1, len(validation)))
            sys.stdout.flush()

            src = os.path.join(train_dir, file)
            src_mask = os.path.join(train_mask_dir, file)

            folder = "./data/val_while_train/"
            mask_folder = "./data/val_mask_while_train/"

            if not os.path.exists(folder):
                os.makedirs(folder)
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            dst = os.path.join(folder, file)
            dst_mask = os.path.join(mask_folder, file)

            shutil.copyfile(src, dst)
            shutil.copyfile(src_mask, dst_mask)
        print("\n   Finish generate validation data")
