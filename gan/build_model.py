
import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import time

from export_images import import_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMG_ROWS = 64
IMG_COLS = 64
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Change these!!
TRAINER = 'jon'
BATCH_SIZE = 32
EPOCHS = 1
MODELNAME = f"{TRAINER}/b{BATCH_SIZE}-e{EPOCHS}.h5"

CATEGORY = "Dog"
PATH_TO_IMAGES = f"{os.getcwd()}/gan/Images/{CATEGORY}"

# Layer constants
noise_dim = 128
noise_shape = (noise_dim,)
leaky_relu_slope = 0.1
weight_init_std = 0.02
weight_init_mean = 0
dropout_rate = 0.4
#weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=weight_init_std, mean=weight_init_mean, seed=42)

def build_generator(img_shape):

    image_height = img_shape[0]
    image_width = img_shape[1]
    image_channels = img_shape[2]

    model = Sequential()

    model.add(Dense(16 * 16 * 128, activation='relu', input_shape=noise_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(Reshape((16, 16, 128)))

    model.add(Conv2D(128, 5, strides=1,padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2DTranspose(128, 4, strides=2,padding='same'))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2D(128, 5, strides=1,padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2D(128, 5, strides=1,padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2D(3, 5, strides=1,padding='same', activation='tanh'))
    
    #model.summary()

    

    noise = Input( shape=noise_shape )
    img = model(noise)

    return Model(noise, img)

def build_discriminator(img_shape):

    model = Sequential()

    model.add(Conv2D(128, 3, strides=1, padding='same', input_shape=img_shape))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2D(128, 4, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2D(128, 4, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Conv2D(128, 4, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(LeakyReLU(alpha=leaky_relu_slope))

    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    #model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def train(n_epochs, batch_size, generator, discriminator, combined_model, image_data):

    print ("Starting training with %d epochs and batch size %d" % (n_epochs, batch_size))
    half_batch = int(batch_size/2)

    for epoch in range(1,n_epochs + 1):
        time_now = time.time()
        idx = np.random.randint(0, image_data.shape[0], half_batch)
        imgs = image_data[idx]


        noise = np.random.normal(0, 1, (half_batch, noise_dim))
        gen_imgs = generator.predict(noise)
        gen_imgs += 1
        gen_imgs /= 2

        # Train discriminator on real images, then generated (fake), labeled 1 and 0 respectivly
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # Train generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))

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
    z = Input(shape=noise_shape)
    img = generator(z)
    
    # In the combined model, only the generator will be trained
    discriminator.trainable = False 

    valid = discriminator(img)


    # The combined model
    combined_model = Model(z, valid)
    combined_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Import images, convert to float [0 1]
    image_data = import_images(PATH_TO_IMAGES, (IMG_ROWS,IMG_COLS), n_images=1000) / 255
    image_data = image_data[:,:,:,::-1] # BRG -> RGB

    # Train generator and discriminator together
    train(EPOCHS, BATCH_SIZE, generator, discriminator, combined_model, image_data)

    # Generate image from model
    noise = np.random.normal(0,1,(1,noise_dim))

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