from tensorflow.keras.layers import Conv2D,Dense,Dropout,BatchNormalization,Flatten,MaxPool2D,Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image

def generate_classifier(size=128,classes = 4,learning_rate=0.01):
    x,y = size,size#pixel_crop*number_of_crops,pixel_crop*number_of_crops
    i = Input(shape=(x,y,3))
    m = Conv2D(64,kernel_size=(3,3),padding="same",activation="relu")(i)
    m = BatchNormalization()(m)
    m = MaxPool2D()(m)
    m = Conv2D(64,kernel_size=(3,3),padding="same",activation="relu")(m)
    m = BatchNormalization()(m)
    m = MaxPool2D()(m)
    m = Conv2D(128,kernel_size=(3,3),padding="same",activation="relu")(m)
    m = BatchNormalization()(m)
    m = MaxPool2D()(m)
    m = Conv2D(256,kernel_size=(3,3),padding="same",activation="relu")(m)
    m = BatchNormalization()(m)
    m = MaxPool2D()(m)
    m = Conv2D(512,kernel_size=(3,3),padding="same",activation="relu")(m)
    m = BatchNormalization()(m)
    m = MaxPool2D()(m)
    m = Flatten()(m)
    m = Dense(512,activation="relu")(m)
    m = Dropout(0.3)(m)
    out1 = Dense(classes,activation="softmax",name="output_classes")(m)
    out2 = Dense(1,activation="sigmoid",name="output_score")(m)
    m = Model(i,[out1,out2])
    optimizer = Adam(learning_rate=learning_rate)
    m.compile(optimizer=optimizer,
                  loss={'output_score': 'mse', 'output_classes': 'categorical_crossentropy'},
                  loss_weights={'output_score': 5., 'output_classes': 1.})
    return m