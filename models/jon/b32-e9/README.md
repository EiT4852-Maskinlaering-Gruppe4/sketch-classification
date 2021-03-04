
# Model

```python
BATCHES = 32
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
10557/10557 [==============================] - 47s 4ms/step - loss: 0.9929 - accuracy: 0.6569 - val_loss: 0.2943 - val_accuracy: 0.8953
Epoch 2/9
10557/10557 [==============================] - 47s 4ms/step - loss: 0.2784 - accuracy: 0.8995 - val_loss: 0.2607 - val_accuracy: 0.9059
Epoch 3/9
10557/10557 [==============================] - 48s 5ms/step - loss: 0.2400 - accuracy: 0.9123 - val_loss: 0.2333 - val_accuracy: 0.9161
Epoch 4/9
10557/10557 [==============================] - 47s 4ms/step - loss: 0.2225 - accuracy: 0.9184 - val_loss: 0.2305 - val_accuracy: 0.9171
Epoch 5/9
10557/10557 [==============================] - 48s 5ms/step - loss: 0.2085 - accuracy: 0.9231 - val_loss: 0.2226 - val_accuracy: 0.9193
Epoch 6/9
10557/10557 [==============================] - 48s 5ms/step - loss: 0.1973 - accuracy: 0.9270 - val_loss: 0.2170 - val_accuracy: 0.9223
Epoch 7/9
10557/10557 [==============================] - 47s 4ms/step - loss: 0.1864 - accuracy: 0.9307 - val_loss: 0.2262 - val_accuracy: 0.9196
Epoch 8/9
10557/10557 [==============================] - 47s 4ms/step - loss: 0.1801 - accuracy: 0.9326 - val_loss: 0.2203 - val_accuracy: 0.9207
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 24, 24, 20)        520       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 20)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 32)          16032     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 32)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 700)               359100    
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 3505      
=================================================================
Total params: 379,157
Trainable params: 379,157
Non-trainable params: 0
```
