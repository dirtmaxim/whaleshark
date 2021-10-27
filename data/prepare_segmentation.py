import os
import json
import imageio
import urllib.request
import numpy as np
import cv2
from tqdm import tqdm
import random

# Create segmentation dataset.
if __name__ == "__main__":
    with open("whalesharkJSON.json") as file:
        data = json.load(file)

    if not os.path.exists("dataset_segmentation"):
        os.makedirs("dataset_segmentation")

    min_size = 512
    counter = 0

    for encounter in tqdm(data["encounters"]):
        if "leftReferenceImageURL" in encounter.keys():
            url = encounter["leftReferenceImageURL"]
            spots = encounter["leftSpots"]
            position = "left"
        else:
            url = encounter["rightReferenceImageURL"]
            spots = encounter["rightSpots"]
            position = "right"

        try:
            url = url.replace(" ", "%20")

            if url.find(".gif") > -1:
                image = np.array(imageio.imread(url))
                image = image[:, :, :3]
                image = image[:, :, ::-1]
            else:
                request = urllib.request.urlopen(url, timeout=100)
                image = cv2.imdecode(np.asarray(bytearray(request.read()), dtype=np.uint8), -1)

            old_height = image.shape[0]
            old_width = image.shape[1]

            if image.shape[0] > image.shape[1]:
                new_height = int(image.shape[0] * min_size / image.shape[1])
                new_width = min_size
                image = cv2.resize(image, (new_width, new_height))
            else:
                new_height = min_size
                new_width = int(image.shape[1] * min_size / image.shape[0])
                image = cv2.resize(image, (new_width, new_height))

            spot_locations = []

            for spot in spots:
                x = int(float(spot["x"]) / old_width * new_width)
                y = int(float(spot["y"]) / old_height * new_height)
                spot_locations.append([x, y])

            if encounter["individualID"] != "":
                individual_id = encounter["individualID"]
            else:
                individual_id = "ID_NOT_ASSIGNED"

            path = "dataset_segmentation" + os.sep + individual_id

            if not os.path.exists(path):
                os.makedirs(path)

            mask = np.zeros(shape=(image.shape[0], image.shape[1]), dtype=np.uint8)

            for spot_location in spot_locations:
                radius = int(0.02 * mask.shape[0])
                cv2.circle(mask, (spot_location[0], spot_location[1]), radius, 255, -1)

            cv2.imwrite(path + os.sep + str(counter) + "_{0}".format(position) + ".png", image)
            cv2.imwrite(path + os.sep + str(counter) + "_{0}_spots".format(position) + ".png", mask)
            counter += 1
        except Exception as e:
            print(e)

    dirs = os.listdir("dataset_segmentation")
    dirs.remove("ID_NOT_ASSIGNED")
    random.shuffle(dirs)
    train_dirs = dirs[:int(len(dirs) * 0.65)]
    train_dirs += ["ID_NOT_ASSIGNED"]
    val_dirs = dirs[int(len(dirs) * 0.65):int(len(dirs) * 0.75)]
    test_dirs = dirs[int(len(dirs) * 0.75):]
    total_size = 0
    train_size = 0
    val_size = 0
    test_size = 0

    for entry in os.listdir("dataset_segmentation"):
        total_size += len(os.listdir("dataset_segmentation" + os.sep + entry))

    total_size += len(os.listdir("dataset_segmentation" + os.sep + "ID_NOT_ASSIGNED"))

    for entry in train_dirs:
        train_size += len(os.listdir("dataset_segmentation" + os.sep + entry))

    for entry in val_dirs:
        val_size += len(os.listdir("dataset_segmentation" + os.sep + entry))

    for entry in test_dirs:
        test_size += len(os.listdir("dataset_segmentation" + os.sep + entry))

    print("Total: {0}.".format(total_size // 2))
    print("Train: {0}.".format(train_size // 2))
    print("Val: {0}.".format(val_size // 2))
    print("Test: {0}.".format(test_size // 2))
