import os
import cv2
import json
from tqdm import tqdm

# https://lila.science/datasets/whale-shark-id
# Create gallery.
if __name__ == "__main__":
    with open("whaleshark.coco/instances_train2020.json") as file:
        data = json.load(file)

    if not os.path.exists("dataset_identification"):
        os.makedirs("dataset_identification")

    id2name = dict()

    for entry in data["images"]:
        id2name[entry["id"]] = entry["file_name"]

    categories = dict()
    counter = 0

    for entry in data["annotations"]:
        individual_ids = tuple(sorted(entry["individual_ids"]))

        if categories.get(individual_ids) is None:
            categories[individual_ids] = counter
            counter += 1

    for entry in tqdm(data["annotations"]):
        individual_ids = tuple(sorted(entry["individual_ids"]))
        image_id = entry["image_id"]
        file_name = id2name[image_id]
        image = cv2.imread("whaleshark.coco/images/{0}".format(file_name))
        x, y, width, height = entry["bbox"]
        crop = image[int(y):int(y) + int(height), int(x):int(x) + int(width)]
        path = "dataset_identification" + os.sep + str(categories[individual_ids])

        if not os.path.exists(path):
            os.makedirs(path)

        cv2.imwrite(path + os.sep + file_name, crop)
