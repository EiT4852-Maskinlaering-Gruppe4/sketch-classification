# Model

```python
# Model definition
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(40, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Dense(700, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(350, activation='relu'))
model.add(Dense(6, activation='softmax'))

early_stop = EarlyStopping(monitor="val_loss",
                           min_delta=0,
                           patience=2,
                           verbose=0,
                           mode="auto",
                           baseline=None,
                           restore_best_weights=True)

model.compile(optimizer=SGD(lr=0.001, momentum=0.8, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Load training, test set.

train_set, test_set, train_labels, test_labels, label_strings = load_sketches(0.7, "./data/")
train_set = np.reshape(train_set, (train_set.shape[0], 28, 28, 1))


# Trains the model and saves the history of the training
history = model.fit(train_set, train_labels, batch_size=64, epochs=4, validation_split=0.3, callbacks=[early_stop])

model.summary()
```

# Summary

```bash
Epoch 1/4
5279/5279 [==============================] - 130s 25ms/step - loss: 0.7803 - accuracy: 0.6896 - val_loss: 0.4279 - val_accuracy: 0.8682
Epoch 2/4
5279/5279 [==============================] - 115s 22ms/step - loss: 0.4567 - accuracy: 0.8282 - val_loss: 0.4628 - val_accuracy: 0.8802
Epoch 3/4
5279/5279 [==============================] - 116s 22ms/step - loss: 0.4012 - accuracy: 0.8491 - val_loss: 0.4617 - val_accuracy: 0.8937
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_18 (Conv2D)           (None, 24, 24, 20)        520
_________________________________________________________________
max_pooling2d_18 (MaxPooling (None, 12, 12, 20)        0
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 8, 8, 32)          16032
_________________________________________________________________
max_pooling2d_19 (MaxPooling (None, 4, 4, 32)          0
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 2, 2, 40)          11560
_________________________________________________________________
max_pooling2d_20 (MaxPooling (None, 1, 1, 40)          0
_________________________________________________________________
dropout_12 (Dropout)         (None, 1, 1, 40)          0
_________________________________________________________________
flatten_6 (Flatten)          (None, 40)                0
_________________________________________________________________
dense_18 (Dense)             (None, 700)               28700
_________________________________________________________________
dropout_13 (Dropout)         (None, 700)               0
_________________________________________________________________
dense_19 (Dense)             (None, 350)               245350
_________________________________________________________________
dense_20 (Dense)             (None, 6)                 2106
=================================================================
Total params: 304,268
Trainable params: 304,268
Non-trainable params: 0
_________________________________________________________________
```
