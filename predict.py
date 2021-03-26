###############################################
# generate segmentation results
###############################################
from keras.models import *
from keras.callbacks import *
import pickle as pkl
import argparse
import data_generate
import h5py
import glob
import re
from metrics import tversky_loss, focal_loss, dice_loss, jacc_loss, jacc_coef, dice_coef, dice_coefficient_loss

height = 512
width = 512
use_hsv = False

parser = argparse.ArgumentParser()

parser.add_argument("--predict", type=bool, default=True)
parser.add_argument("--ensemble", type=str, default=False)
parser.add_argument("--threshold", type=float, default=0.5)

parser.add_argument("--model_path", type=str, default="./trainedModel")

parser.add_argument("--test_folder", type=str,
                    default="./datasets9/test/")
parser.add_argument("--test_mask_folder", type=str,
                    default="./datasets9/test_mask/")
parser.add_argument("--test_resized_folder", type=str,
                    default="./datasets9/test/")
#                    default="./data_{}_{}".format(height, width) + "./datasets9/test/")

parser.add_argument("--test_predicted_folder", type=str, default="./Test_Predicted")

parser.add_argument("--remove_mean_imagenet", type=bool, default=True)

model_name = "model1"
ensemble_pkl_filenames = ["model1", "model2", "model3", "model4"]


def predict_challenge(challenge_folder, challenge_mask_folder, challenge_predicted_folder, resized_forder, mask_pred_challenge=None, plot=True):
    challenge_list = data_generate.list_from_folder(challenge_folder)
    challenge_images = data_generate.load_images(challenge_list, height, width,
                                                 image_folder=resized_forder,
                                                 mask_folder=None,
                                                 remove_mean_imagenet=True,
                                                 use_hsv=use_hsv,
                                                 )

    if mask_pred_challenge is None: 
        mask_pred_challenge = model.predict(challenge_images, batch_size=1)
    if not os.path.exists(challenge_predicted_folder):
        os.makedirs(challenge_predicted_folder)
    with open('{}.pkl'.format(os.path.join(challenge_predicted_folder, model_name)), 'wb') as f_pkl:
        pkl.dump(mask_pred_challenge, f_pkl)

    mask_pred_challenge = np.where(mask_pred_challenge >= 0.5, 1, 0)
    mask_pred_challenge = mask_pred_challenge * 255
    mask_pred_challenge = mask_pred_challenge.astype(np.uint8)
    challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name)
    if not os.path.exists(challenge_predicted_folder):
        os.makedirs(challenge_predicted_folder)

    print("Start challenge prediction:")
  
    for i in range(len(challenge_list)):
        print("{}: {}".format(i, challenge_list[i]))
        data_generate.show_images_full_sized(image_list=challenge_list,
                                             img_mask_pred_array=mask_pred_challenge,
                                             image_folder=challenge_folder,
                                             mask_folder=challenge_mask_folder,
                                             index=i,
                                             output_folder=challenge_predicted_folder,
                                             plot=plot
                                             )


def join_predictions(pkl_folder, pkl_files, binary=False, threshold=0.5):
    n_pkl = float(len(pkl_files))
    array = None
    for fname in pkl_files:
        with open(os.path.join(pkl_folder, fname + ".pkl"), "rb") as f_pkl:
            tmp = pkl.load(f_pkl)
            if binary:
                tmp = np.where(tmp >= threshold, 1, 0)
            if array is None:
                array = tmp
            else:
                array = array + tmp
    return array / n_pkl


def print_keras_weights(weight_file_path):
    file = h5py.File(weight_file_path)
    if len(file.attrs.items()):
        print("{} contains: ".format(weight_file_path))
        print("Root attributes:")
    for key, value in file.attrs.items():
        print("  {}: {}".format(key, value))
    for layer, g in file.items():
        print("  {}".format(layer))
        print("    Attributes:")
        for key, value in g.attrs.items():
            print("      {}: {}".format(key, value))

    file.close()


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    print('Load model')
    model_files = glob.glob(FLAGS.model_path + "/*.hdf5")
    assert len(model_files) != 0, 'There is no model to load!'

    if len(model_files) == 1:
        model_file = model_files[0]
    else:
        pre_loss = float('inf')
        model_file = ""
        for file in model_files:
            values = re.findall("\d+\.\d+", file)
            cur_loss = eval(values[0])
            if cur_loss < pre_loss:
                pre_loss = cur_loss
                model_file = file

    model = load_model(model_file, custom_objects={'dice_loss': dice_loss,
                                                   'dice_coef': dice_coef,
                                                   'jacc_loss': jacc_loss,
                                                   'jacc_coef': jacc_coef,
                                                   'tversky_loss': tversky_loss,
                                                   'focal_loss': focal_loss,
                                                   'dice_coefficient_loss': dice_coefficient_loss})
    if FLAGS.predict:
        print("Start Challenge Test")
        predict_challenge(challenge_folder=FLAGS.test_folder,
                          challenge_mask_folder=FLAGS.test_mask_folder,
                          challenge_predicted_folder=FLAGS.test_predicted_folder,
                          resized_forder=FLAGS.test_resized_folder,
                          plot=False)
    if FLAGS.ensemble:
        test_array = join_predictions(pkl_folder=FLAGS.test_predicted_folder,
                                      pkl_files=ensemble_pkl_filenames,
                                      binary=False,
                                      threshold=FLAGS.threshold
                                      )

        model_name = "ensemble_{}".format(FLAGS.threshold)
        for f in ensemble_pkl_filenames:
            model_name = model_name + "_" + f
        print("Predict Test:")
        predict_challenge(challenge_folder=FLAGS.test_folder,
                          challenge_mask_folder=FLAGS.test_mask_folder,
                          challenge_predicted_folder=FLAGS.test_predicted_folder,
                          mask_pred_challenge=test_array,
                          resized_forder=FLAGS.test_resized_folder,
                          plot=False
                          )
