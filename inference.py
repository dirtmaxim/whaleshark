import os
import sys
import numpy as np
import cv2
import segmentation_models as sm
from configparser import ConfigParser
from transformations import transform_test

sm.set_framework("tf.keras")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class SpotDetector:
    def __init__(self):
        with open("config.cfg", "r") as file:
            config = ConfigParser()
            config.read_file(file)

        self.shape = config["Parameters"].getint("shape")
        self.f0_model = sm.Unet(config["Parameters"]["backbone"], encoder_weights="imagenet")
        self.f0_model.load_weights("models/fold_0/best.h5")
        self.f1_model = sm.Unet(config["Parameters"]["backbone"], encoder_weights="imagenet")
        self.f1_model.load_weights("models/fold_1/best.h5")
        self.f2_model = sm.Unet(config["Parameters"]["backbone"], encoder_weights="imagenet")
        self.f2_model.load_weights("models/fold_2/best.h5")
        self.f3_model = sm.Unet(config["Parameters"]["backbone"], encoder_weights="imagenet")
        self.f3_model.load_weights("models/fold_3/best.h5")

    def __tta(self, image_, model_):
        flipped = np.fliplr(image_)
        predicted_mask_ = model_.predict(np.expand_dims(image_, axis=0))
        predicted_flipped = model_.predict(np.expand_dims(flipped, axis=0))
        predicted_mask_ = np.squeeze(predicted_mask_)
        predicted_reverse = np.fliplr(np.squeeze(predicted_flipped))

        return (predicted_mask_ + predicted_reverse) / 2

    def predict(self, image):
        transformed = transform_test(image=image)
        image = transformed["image"].astype(np.float32) / 255
        predicted = []

        for model in [self.f0_model, self.f1_model, self.f2_model, self.f3_model]:
            predicted.append(self.__tta(image, model))

        predicted = np.mean(predicted, axis=0)
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        predicted = cv2.erode(predicted, kernel, iterations=1)
        predicted = cv2.dilate(predicted, kernel, iterations=1)
        predicted_uint8 = (predicted * 255).astype(np.uint8)
        contours_pred, _ = cv2.findContours(predicted_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []

        for contour in contours_pred:
            M = cv2.moments(contour)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            points.append([x, y])

        return transformed["image"], predicted_uint8, points


# Usage: python inference.py path_to_image
if __name__ == "__main__":
    sd = SpotDetector()
    image = cv2.imread(sys.argv[1])
    transformed_image, mask, points = sd.predict(image)
    blended = cv2.addWeighted(transformed_image, 0.5, np.stack([mask, mask, mask], axis=-1), 0.5, 0)
    print("Point Locations:")
    print(points)
    cv2.imshow("Result", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
