import cv2
import albumentations as A
from configparser import ConfigParser

with open("config.cfg", "r") as file:
    config = ConfigParser()
    config.read_file(file)

shape = config["Parameters"].getint("shape")

transform_train = A.Compose([
    A.OneOf([
        A.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, rotate=(-15, 15), shear=(-10, 10), p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, shear=(-10, 10), p=0.5),
    ], p=0.5),
    A.LongestMaxSize(max_size=shape, p=1.0),
    A.PadIfNeeded(min_height=shape, min_width=shape, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2)
])
transform_test = A.Compose([
    A.LongestMaxSize(max_size=shape, p=1.0),
    A.PadIfNeeded(min_height=shape, min_width=shape, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)
])
