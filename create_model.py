from load_sketches import load_sketches
import numpy as np
import tensorflow as tf
import tensorflow.keras


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from matplotlib import pyplot as plt

# REMEMBER TO CHANGE THESE
MODELNAME = 'ralf/b200-e8'
PATH_TO_DATA = './sketches/'

BATCHES = 200
EPOCHS = 9

# Model definition
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(700, activation='relu'))
#model.add(Dropout(0.3))
#model.add(Dense(350, activation='relu'))
model.add(Dense(5, activation='softmax'))

early_stop = EarlyStopping(monitor="val_loss",
                           min_delta=0,
                           patience=2,
                           verbose=0,
                           mode="auto",
                           baseline=None,
                           restore_best_weights=True)

model.compile(optimizer=SGD(lr=0.001, momentum=0.8, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Load training, test set.

train_set, test_set, train_labels, test_labels, label_strings = load_sketches(0.7, PATH_TO_DATA)
train_set = np.reshape(train_set, (train_set.shape[0], 28, 28, 1))


# Trains the model and saves the history of the training
history = model.fit(train_set, train_labels, batch_size=BATCHES, epochs=EPOCHS, validation_split=0.3, callbacks=[early_stop])

model.summary()

model.save('./models/' + MODELNAME)
