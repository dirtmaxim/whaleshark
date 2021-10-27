import os
import random
import segmentation_models as sm
import cv2
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
from configparser import ConfigParser
from transformations import transform_test
from metrics import dice_coeff, iou

sm.set_framework("tf.keras")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def tta(image_, model_):
    flipped = np.fliplr(image_)
    predicted_mask_ = model_.predict(np.expand_dims(image, axis=0))
    predicted_flipped = model_.predict(np.expand_dims(flipped, axis=0))
    predicted_mask_ = np.squeeze(predicted_mask_)
    predicted_reverse = np.fliplr(np.squeeze(predicted_flipped))

    return (predicted_mask_ + predicted_reverse) / 2


if __name__ == "__main__":
    with open("config.cfg", "r") as file:
        config = ConfigParser()
        config.read_file(file)

    backbone = config["Parameters"]["backbone"]
    seed = config["Parameters"].getint("seed")
    shape = config["Parameters"].getint("shape")
    dataset_path = config["Parameters"]["dataset_path"]
    batch_size = config["Parameters"].getint("batch_size")
    epochs = config["Parameters"].getint("epochs")
    predicted_path = config["Parameters"]["predicted_path"]
    random.seed(seed)

    # Get dataset as list of paths.
    image_files = []
    mask_files = []

    for root, dirs, files in sorted(os.walk("data/dataset_segmentation")):
        for file in files:
            filename, extension = os.path.splitext(file)

            if filename[-6:] != "_spots":
                if extension in [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png"]:
                    image_files.append(os.path.join(root, file))
                    mask_files.append(os.path.join(root, filename + "_spots" + extension))

    image_files = np.array(image_files)
    mask_files = np.array(mask_files)

    # Create folds model.
    folds_model = KFold(n_splits=4)

    mean_dice = []
    mean_iou = []

    for fold_id, (train_index, test_index) in enumerate(folds_model.split(image_files, mask_files)):
        fold = "models/fold_{0}".format(fold_id)
        test_images, test_masks = image_files[test_index], mask_files[test_index]
        model = sm.Unet(backbone, encoder_weights="imagenet")
        model.load_weights("{0}/best.h5".format(fold))
        dices = []
        ious = []

        for i, (image_path, mask_path) in tqdm(enumerate(zip(test_images, test_masks)), total=len(test_images)):
            output_image = predicted_path + os.sep + "/".join(image_path.split("/")[-2:])
            output_mask = predicted_path + os.sep + "/".join(mask_path.split("/")[-2:])
            filename, extension = os.path.splitext(image_path.split("/")[-1])
            predicted_mask = predicted_path + os.sep + image_path.split("/")[
                -2] + os.sep + filename + "_predicted" + extension

            if not os.path.exists(predicted_path + os.sep + image_path.split("/")[-2]):
                os.makedirs(predicted_path + os.sep + image_path.split("/")[-2])

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, 0)
            transformed = transform_test(image=image, mask=mask)
            image = transformed["image"].astype(np.float32) / 255
            mask = transformed["mask"].astype(np.float32) / 255
            predicted = tta(image, model)
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            predicted = cv2.erode(predicted, kernel, iterations=1)
            predicted = cv2.dilate(predicted, kernel, iterations=1)
            cv2.imwrite(output_image, (image * 255).astype(np.uint8))
            cv2.imwrite(output_mask, (mask * 255).astype(np.uint8))
            cv2.imwrite(predicted_mask, (predicted * 255).astype(np.uint8))
            dice_score = dice_coeff(mask, predicted)
            iou_score = iou(mask, predicted)
            dices.append(dice_score)
            ious.append(iou_score)
            mean_dice.append(dice_score)
            mean_iou.append(iou_score)

        print("Fold {0} dice: {1:.3f}".format(fold_id, np.mean(dices)))
        print("Fold {0} IoU: {1:.3f}".format(fold_id, np.mean(ious)))

    print("Dice: {0:.3f}".format(np.mean(mean_dice)))
    print("IoU: {0:.3f}".format(np.mean(mean_iou)))
