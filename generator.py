import numpy as np
import cv2
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images_list, masks_list, transform, batch_size, shape, shuffle=True):
        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images_list) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_images_list = [self.images_list[i] for i in indexes]
        temp_masks_list = [self.masks_list[i] for i in indexes]
        x, y = self.__data_generation(temp_images_list, temp_masks_list)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_list))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_images_list, temp_masks_list):
        x = np.zeros(shape=(len(temp_images_list), self.shape, self.shape, 3), dtype=np.float32)
        y = np.zeros(shape=(len(temp_masks_list), self.shape, self.shape), dtype=np.float32)

        for i, (image_path, mask_path) in enumerate(zip(temp_images_list, temp_masks_list)):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, 0)
            transformed = self.transform(image=image, mask=mask)
            x[i] = transformed["image"].astype(np.float32) / 255
            y[i] = transformed["mask"].astype(np.float32) / 255

        return x, y
