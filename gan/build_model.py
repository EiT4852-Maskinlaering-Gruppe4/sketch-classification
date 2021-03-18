
import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import time

from export_images import import_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_ROWS = 100
IMG_COLS = 100
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

MODELNAME = 'jon/b128-e50'
BATCH_SIZE = 256
EPOCHS = 1

CATEGORY = "Dog"
PATH_TO_IMAGES = f"{os.getcwd()}/gan/Images/{CATEGORY}"

def build_generator(img_shape):

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

def build_discriminator(img_shape):


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

def train(n_epochs, batch_size, generator, discriminator, combined_model, image_data):

    half_batch = int(batch_size/2)

    for epoch in range(1,n_epochs + 1):
        time_now = time.time()
        idx = np.random.randint(0, image_data.shape[0], half_batch)
        imgs = image_data[idx]


        noise = np.random.normal(0, 1, (half_batch, 100))
        gen_imgs = generator.predict(noise)
        gen_imgs += 1
        gen_imgs /= 2

        # Train discriminator on real images, then generated (fake), labeled 1 and 0 respectivly
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))

        # The generator wants to train for 100% validity, ie. 1
        valid_y = np.array([1] * batch_size)


        g_loss = combined_model.train_on_batch(noise, valid_y)
        
        
        duration = time.time() - time_now
        print ("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f] Duration: %.2f" % (epoch, d_loss[0], 100*d_loss[1], g_loss, duration))


def main():
    optimizer = Adam(0.0002, 0.5) # Learning rate, momentum


    discriminator = build_discriminator(IMG_SHAPE)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator(IMG_SHAPE)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)


    # 
    z = Input(shape=(100,))
    img = generator(z)

    # In the combined model, only the generator will be trained
    discriminator.trainable = False 

    valid = discriminator(img)


    # The combined model
    combined_model = Model(z, valid)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Import images, convert to float [0 1]
    image_data = import_images(PATH_TO_IMAGES, (IMG_ROWS,IMG_COLS), n_images=5000) / 255
    image_data = image_data[:,:,:,::-1] # BRG -> RGB

    # Train generator and discriminator together
    train(EPOCHS, BATCH_SIZE, generator, discriminator, combined_model, image_data)

    # Generate image from model
    noise = np.random.normal(0,1,(1,100))

    gen_image = generator.predict(noise)
    gen_image += 1
    gen_image /= 2
    #print(gen_image[0][0])


    plt.imshow(gen_image[0])
    plt.show()

    # Save the generator, discard the dicriminator
    generator.save('./models/' + MODELNAME)

if __name__ == "__main__":
    main()