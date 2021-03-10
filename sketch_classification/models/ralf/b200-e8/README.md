
# Model

```python
BATCHES = 200
EPOCHS = 9

# Model definition
model = Sequential()
model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(700, activation='relu'))
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
```

# Summary
```bash
Epoch 1/9
1689/1689 [==============================] - 93s 54ms/step - loss: 2.1497 - accuracy: 0.3817 - val_loss: 0.7211 - val_accuracy: 0.7410
Epoch 2/9
1689/1689 [==============================] - 92s 55ms/step - loss: 0.6505 - accuracy: 0.7621 - val_loss: 0.5369 - val_accuracy: 0.8003
Epoch 3/9
1689/1689 [==============================] - 88s 52ms/step - loss: 0.4712 - accuracy: 0.8277 - val_loss: 0.3653 - val_accuracy: 0.8693
Epoch 4/9
1689/1689 [==============================] - 89s 53ms/step - loss: 0.3500 - accuracy: 0.8709 - val_loss: 0.3205 - val_accuracy: 0.8829
Epoch 5/9
1689/1689 [==============================] - 90s 53ms/step - loss: 0.3022 - accuracy: 0.8901 - val_loss: 0.2814 - val_accuracy: 0.8996
Epoch 6/9
1689/1689 [==============================] - 89s 53ms/step - loss: 0.2738 - accuracy: 0.9007 - val_loss: 0.2675 - val_accuracy: 0.9061
Epoch 7/9
1689/1689 [==============================] - 90s 53ms/step - loss: 0.2596 - accuracy: 0.9066 - val_loss: 0.2582 - val_accuracy: 0.9083
Epoch 8/9
1689/1689 [==============================] - 91s 54ms/step - loss: 0.2465 - accuracy: 0.9113 - val_loss: 0.2665 - val_accuracy: 0.9043
Epoch 9/9
1689/1689 [==============================] - 88s 52ms/step - loss: 0.2380 - accuracy: 0.9145 - val_loss: 0.2496 - val_accuracy: 0.9103
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
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 700)               359100    
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 3505      
=================================================================
Total params: 379,157
Trainable params: 379,157
Non-trainable params: 0
```