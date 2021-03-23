import os
import pdb
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3" # 使用第三、四块GPU
import numpy as np
from metrics import tversky_loss, focal_loss, dice_loss, jacc_loss, jacc_coef, dice_coef, dice_jacc_mean, precision, dice_coefficient_loss, \
    sensitivity, specificity
#from metrics import dice_loss, jacc_loss, jacc_coef, dice_coef, dice_jacc_mean, precision, \
#    sensitivity, specificity
import data_generate
import argparse
import glob
import re
import keras


model_path = "./trainedModel/"
#test_dir = "./data_128_128/test_image/"
#test_mask_dir = "./data_128_128/test_mask/"

#test_dir = "./ACDC_on_128/test_image/"
#test_mask_dir = "./ACDC_on_128/test_mask/"

#test_dir = "./miccai_on_128/test_image/"
#test_mask_dir = "./miccai_on_128/test_mask/"

#test_dir = "./all_on_128/test_image/"
#test_mask_dir = "./all_on_128/test_mask/"

#test_dir= "./plan10/test/data52/iniimages/"
#test_mask_dir="./plan10/test/data52/mask/"

#test_dir= "./plan14/test/"
#test_mask_dir="./plan14/test_mask/"


#test_dir= "./plan16/test/"
#test_mask_dir="./plan16/test_mask/"
test_dir= "./plan14(paper)/test/"
test_mask_dir="./plan14(paper)/test_mask/"
parser = argparse.ArgumentParser()
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--remove_mean_imagenet", type=bool, default=True)
parser.add_argument("--rescale_mask", type=bool, default=True)
parser.add_argument("--use_hsv", type=bool, default=False)

def run_test(FLAGS):
    test_list = os.listdir(test_dir)
    print(">> Test...\n>> Loading images...")
    test, test_mask = data_generate.load_images(test_list, FLAGS.height, FLAGS.width,
                                                test_dir, test_mask_dir,
                                                remove_mean_imagenet=FLAGS.remove_mean_imagenet,
                                                rescale_mask=FLAGS.rescale_mask, use_hsv=FLAGS.use_hsv
                                                )
    print(">> Loading images done")
    print(">> Loading best checkpoint to test")
    checkpoint_files = glob.glob(model_path + "weights*.hdf5")
    assert len(checkpoint_files) != 0, 'There is no model to load!'
    if len(checkpoint_files) == 1:
        model_file = checkpoint_files[0]
    else:
        model_file = ""
        pre_loss = float('inf')
        for file in checkpoint_files:
            values = re.findall("\d+\.\d+", file)
            cur_loss = eval(values[1])
            if cur_loss < pre_loss:
                pre_loss = cur_loss
                model_file = file
    model = keras.models.load_model(model_file, custom_objects={'dice_loss': dice_loss,
                                                                'dice_coef': dice_coef,
                                                                'jacc_loss': jacc_loss,
                                                                'jacc_coef': jacc_coef,
                                                                'tversky_loss': tversky_loss,
                                                                'focal_loss': focal_loss,
                                                                'dice_coefficient_loss': dice_coefficient_loss})      
    
    print(">> Start Test...")
    mask_pred_test = model.predict(test, batch_size=1)
    for pixel_threshold in [0.5]:
        mask_pred_test = np.where(mask_pred_test >= pixel_threshold, 1, 0)
        mask_pred_test = mask_pred_test.astype(np.uint8)
        mask_pred_test = mask_pred_test * 255
        print(model.evaluate(test, test_mask, batch_size=1, verbose=1))
        dice, jacc = dice_jacc_mean(test_mask, mask_pred_test, smooth=0)
        print(model_file)
        print(">> Resized test dice coef      : {:.4f}".format(dice))
        print(">> Resized test jacc coef      : {:.4f}".format(jacc))
        print("\n")

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    run_test(FLAGS)
