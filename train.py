import os
import random
import segmentation_models as sm
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from configparser import ConfigParser
from transformations import transform_train, transform_test
from generator import DataGenerator
from metrics import dice_coeff, loss, iou

sm.set_framework("tf.keras")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def augmentations_example(folds_model_, image_files_, mask_files_, sample_size=8):
    for fold_id_, (train_index_, test_index_) in enumerate(folds_model_.split(image_files_, mask_files_)):
        train_images_, test_images_ = image_files_[train_index_], image_files_[test_index_]
        train_masks_, test_masks_ = mask_files_[train_index_], mask_files_[test_index_]
        train_images_, val_images_, train_masks_, val_masks_ = train_test_split(train_images_, train_masks_,
                                                                                test_size=0.2, random_state=seed)
        data_generator = DataGenerator(images_list=train_images_, masks_list=train_masks_, transform=transform_train,
                                       batch_size=sample_size, shape=shape)
        images, masks = data_generator[0]
        fig, axarr = plt.subplots(sample_size, 2, figsize=(10, 40))

        for i in range(sample_size):
            axarr[i][0].axis("off")
            axarr[i][1].axis("off")
            axarr[i][0].imshow((images[i][:, :, ::-1] * 255).astype(np.uint8))
            axarr[i][1].imshow(masks[i] * 255, cmap="gray")

        plt.savefig("logs/augmentations.png")
        break


def save_plots(history, fold_id):
    plt.clf()
    plt.plot(history.history["dice_coeff"], "b-")
    plt.plot(history.history["val_dice_coeff"], "g-")
    plt.title("Metrics")
    plt.ylabel("dice")
    plt.xlabel("epochs")
    plt.legend(["train", "val"])
    plt.savefig("logs/fold_{0}_dice.png".format(fold_id))
    plt.clf()
    plt.plot(history.history["loss"], "b-")
    plt.plot(history.history["val_loss"], "g-")
    plt.title("Metrics")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train", "val"])
    plt.savefig("logs/fold_{0}_loss.png".format(fold_id))


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

    # Create augmentations example and save it as image file.
    augmentations_example(folds_model, image_files, mask_files)

    # Train 4 models on each fold.
    for fold_id, (train_index, test_index) in enumerate(folds_model.split(image_files, mask_files)):
        fold = "models/fold_{0}".format(fold_id)

        if not os.path.exists(fold):
            os.makedirs(fold)

        train_images, test_images = image_files[train_index], image_files[test_index]
        train_masks, test_masks = mask_files[train_index], mask_files[test_index]
        train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks,
                                                                            test_size=0.2, random_state=seed)

        model = sm.Unet(backbone, encoder_weights="imagenet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss=loss, metrics=[dice_coeff])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(fold + "/epoch_{epoch:02d}.h5", monitor="val_dice_coeff", mode="max",
                                               save_weights_only=True, save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_dice_coeff", factor=0.4, patience=2, verbose=1,
                                                 mode="max", min_lr=0.000000001),
        ]
        train_generator = DataGenerator(images_list=train_images, masks_list=train_masks, transform=transform_train,
                                        batch_size=batch_size, shape=shape)
        val_generator = DataGenerator(images_list=val_images, masks_list=val_masks, transform=transform_test,
                                      batch_size=batch_size, shape=shape, shuffle=False)
        history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)
        save_plots(history, fold_id)
        model.load_weights("{0}/".format(fold) + sorted(os.listdir(fold))[-1])
        model.save_weights("{0}/best.h5".format(fold))
