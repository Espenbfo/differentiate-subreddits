"""
Trains the model on the data in data_folder,
"""

import model
import generator
import os
from tensorflow.keras.callbacks import ModelCheckpoint

DATA_FOLDER="data2"#C:\programmering\differentiate"
BATCH_SIZE= 128
IMAGE_SIZE = 128
LEARNING_RATE =0.0005
LOAD_WEIGHTS=False
EPOCHS = 500
classes = len(os.listdir(DATA_FOLDER))

gen = generator.generator(batch_size=BATCH_SIZE,data_folder=DATA_FOLDER)

folders = next(gen)
with open("weights/categories.txt", "w") as f:
    f.write(str(folders))

total = next(gen)
modl = model.generate_classifier(IMAGE_SIZE,classes,LEARNING_RATE)

if LOAD_WEIGHTS:
    modl.load_weights("weights/accurate.h5")
print(modl.summary())

check1 = ModelCheckpoint(filepath="weights/last.h5",save_best_only=False, save_weights_only=True, verbose=1)
check2 = ModelCheckpoint(filepath="weights/accurate.h5",monitor="loss", save_best_only=True, save_weights_only=True, verbose=1)
modl.fit(gen,batch_size=BATCH_SIZE, steps_per_epoch=total//BATCH_SIZE/10,epochs=500, callbacks=[check1,check2])