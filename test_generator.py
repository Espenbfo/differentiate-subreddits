"""
Generator used for testing, as you may not want to test with
the same amount of classes as when you train
"""

import os
import cv2
import random
import numpy as np
import useful_funcs
data_folder = "data"

def load_image(path):
    if (not useful_funcs.is_jpg(path)):
        return None
    try:
        image = cv2.imread(path)
        image = useful_funcs.pad_to_size(image, 128, 128)
        print(path)
        return image
    except Exception as e:
        print(e)

def generator(batch_size, data_folder=data_folder,total_classes=0):
    batch_img = []
    batch_labels = [[],[]]

    folders = os.listdir(data_folder)
    num_folders = 1
    if total_classes < num_folders:
        total_classes = num_folders
    paths = [os.listdir(data_folder)]
    paths = [[os.path.join(data_folder, p) for p in paths[0]]]

    def randomize(paths):
        for path in paths:
            random.shuffle(path)

    randomize(paths)

    lengths = [len(path) for path in paths]
    current_index = [0]*num_folders
    current_folder = 0

    yield folders
    total = sum(lengths)
    yield total

    while True:
        image = None
        while image is None:
            filename = paths[current_folder][current_index[current_folder]]
            image = load_image(filename)
            current_index[current_folder] += 1
            if current_index[current_folder] == lengths[current_folder]:
                current_index[current_folder] = 0
                random.shuffle(paths[current_folder])

        score = 0.5

        category = [0.]*total_classes
        category[current_folder] = 1.
        category = np.array(category).astype("float32")

        batch_img.append(image)
        batch_labels[0].append(category)
        batch_labels[1].append(score)

        if(len(batch_img) == batch_size):

            batch_img = np.array(batch_img)
            batch_labels[0] = np.array(batch_labels[0])
            batch_labels[1] = np.array(batch_labels[1])

            yield (batch_img,batch_labels)
            batch_img = []
            batch_labels = [[],[]]

        current_index[current_folder] += 1
        if current_index[current_folder] == lengths[current_folder]:
            current_index[current_folder] = 0
            random.shuffle(paths[current_folder])

        current_folder += 1
        if current_folder == num_folders:
            current_folder = 0

if __name__ == "__main__":
    gen = generator(10)

    k = next(gen)
    print(k)
    cv2.imshow("1",k[0][0])
    cv2.waitKey(0)
    cv2.imshow("1",k[0][1])
    cv2.waitKey(0)

    k = next(gen)
    print(k)
    cv2.imshow("1",k[0][0])
    cv2.waitKey(0)