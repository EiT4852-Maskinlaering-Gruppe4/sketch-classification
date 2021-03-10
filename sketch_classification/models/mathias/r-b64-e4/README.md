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

train_set, test_set, train_labels, test_labels, label_strings = load_sketches(0.7, "./data/")
train_set = np.reshape(train_set, (train_set.shape[0], 28, 28, 1))


# Trains the model and saves the history of the training
history = model.fit(train_set, train_labels, batch_size=64, epochs=4, validation_split=0.3, callbacks=[early_stop])
```

# Summary

```bash
Epoch 1/4
5279/5279 [==============================] - 122s 23ms/step - loss: 0.7634 - accuracy: 0.6853 - val_loss: 0.4602 - val_accuracy: 0.8471
Epoch 2/4
5279/5279 [==============================] - 121s 23ms/step - loss: 0.4964 - accuracy: 0.8043 - val_loss: 0.4662 - val_accuracy: 0.8663
Epoch 3/4
5279/5279 [==============================] - 122s 23ms/step - loss: 0.4229 - accuracy: 0.8369 - val_loss: 0.5372 - val_accuracy: 0.8434
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 24, 24, 20)        520
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 20)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 32)          16032
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 4, 4, 32)          0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 2, 40)          11560
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 1, 40)          0
_________________________________________________________________
dropout (Dropout)            (None, 1, 1, 40)          0
_________________________________________________________________
flatten (Flatten)            (None, 40)                0
_________________________________________________________________
dense (Dense)                (None, 700)               28700
_________________________________________________________________
dropout_1 (Dropout)          (None, 700)               0
_________________________________________________________________
dense_1 (Dense)              (None, 350)               245350
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 1755
=================================================================
Total params: 303,917
Trainable params: 303,917
Non-trainable params: 0
_________________________________________________________________
```
