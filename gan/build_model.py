
import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

IMG_ROWS = 28
IMG_COLS = 28
CHANNELS = 3
img_shape = (IMG_ROWS, IMG_COLS, CHANNELS)

MODELNAME = 'jon/b128-e50'
BATCH_SIZE = 128
EPOCHS = 50

def build_generator():

    noise_shape = (100,)

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    #model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)


def build_discriminator():


    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    #model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def train():

    half_batch = BATCH_SIZE/2


    for epoch in range(1,EPOCHS + 1):

        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]


        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)


        # Train discriminator on real images, then generated (fake), labeled 1 and 0 respectivly
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))

        # The generator wants to train for 100% validity, ie. 1
        valid_y = np.array([1] * batch_size)


        g_loss = combined.train_on_batch(noise, valid_y)
        
        
        
        print ("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



# optimizer = Adam(0.0002, 0.5) # Learning rate, momentum


# discriminator = build_discriminator()
# discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# generator = build_generator()
# generator.compile(loss='binary_crossentropy', optimizer=optimizer)


# # 
# z = Input(shape=(100,))
# img = generator(z)

# # In the combined model, only the generator will be trained
# discriminator.trainable = False 

# valid = discriminator(img)


# # The combined model
# combined = Model(z, valid)
# combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# # Train generator and discriminator together
# train()


# # Save the generator, discard the dicriminator
# generator.save('./models/' + MODELNAME)

optimizer = Adam(0.0002, 0.5) # Learning rate, momentum
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

noise = np.random.normal(0,1,(1,100))
gen_image = generator.predict(noise)

print("image: ", gen_image[0])
plt.imshow(gen_image[0], cmap="gray")
plt.show()


discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

acc = discriminator.predict(gen_image)
print("acc: ", acc[0])