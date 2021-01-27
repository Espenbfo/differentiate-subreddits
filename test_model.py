"""
Manual testing of your trained model
"""

import model
import test_generator
import os
import cv2
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

data_folder="D:\data\\filer\\reddit_sub_pasta"
batch_size= 1
image_size = 128
learning_rate =0.005
classes = len(os.listdir("data"))

gen = test_generator.generator(batch_size=batch_size,data_folder=data_folder,total_classes=classes)
foldesr = next(gen)
total = next(gen)
modl = model.generate_classifier(image_size,classes,learning_rate)
modl.load_weights("weights/accurate.h5")

fold = os.listdir("data")

with open("weights/categories.txt", "r") as f:
    folders = f.read()[1:-1]
    folders = folders.split(",")
    folders = [folder.strip()[1:-1] for folder in folders]


output = "result\\session"
os.makedirs(output,exist_ok=True)

for folds in folders:
    os.makedirs(os.path.join(output,folds),exist_ok=True)

index = 0
while True:
    images = next(gen)

    pred = modl.predict(images[0])


    argsort = np.argsort(-pred[0][0])
    argsort = [(folders[arg],pred[0][0][arg]) for arg in argsort]

    max = np.argmax(pred[0][0])
    score = 10**(1/pred[1][0][0])
    cv2.imwrite(os.path.join(output,folders[max],str(pred[0][0][max])[2:]+"_"+str(int(score))+".jpg"),images[0][0])