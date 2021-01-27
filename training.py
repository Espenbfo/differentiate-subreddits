import model
import generator
import os
from tensorflow.keras.callbacks import ModelCheckpoint

data_folder="data2"#C:\programmering\differentiate"
batch_size= 128
image_size = 128
learning_rate =0.0005
load_weights=False
classes = len(os.listdir(data_folder))

gen = generator.generator(batch_size=batch_size,data_folder=data_folder)

folders = next(gen)
with open("weights/categories.txt", "w") as f:
    f.write(str(folders))
total = next(gen)
modl = model.generate_classifier(image_size,classes,learning_rate)
if load_weights:
    modl.load_weights("weights/accurate.h5")
print(modl.summary())
check1 = ModelCheckpoint(filepath="weights/last.h5",save_best_only=False, save_weights_only=True, verbose=1)
check2 = ModelCheckpoint(filepath="weights/accurate.h5",monitor="loss", save_best_only=True, save_weights_only=True, verbose=1)
modl.fit(gen,batch_size=batch_size, steps_per_epoch=total//batch_size/10,epochs=500, callbacks=[check1,check2])