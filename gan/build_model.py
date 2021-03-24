
import numpy as np
import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, LeakyReLU, BatchNormalization, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from SpectralNormalizationKeras import ConvSN2D

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
BATCH_SIZE = 256
EPOCHS = 1
MODELNAME = f"{TRAINER}/b{BATCH_SIZE}-e{EPOCHS}.h5"

CATEGORY = "Dog"
PATH_TO_IMAGES = f"{os.getcwd()}/gan/Images/{CATEGORY}"

# Layer templates
leaky_relu_slope = 1
weight_init_std = 1
weight_init_mean = 1
dropout_rate = 1
weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=weight_init_std, mean=weight_init_mean, seed=42)
def transposed_conv(model, out_channels, ksize, stride_size, ptype='same'):
    model.add(Conv2DTranspose(out_channels, (ksize, ksize),
                              strides=(stride_size, stride_size), padding=ptype, 
                              kernel_initializer=weight_initializer, use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def convSN(model, out_channels, ksize, stride_size):
    model.add(ConvSN2D(out_channels, (ksize, ksize), strides=(stride_size, stride_size), padding='same',
                     kernel_initializer=weight_initializer, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    #model.add(Dropout(dropout_rate))
    return model

def build_generator(img_shape):
    noise_dim = 128

    image_height = img_shape[0]
    image_width = img_shape[1]
    image_channels = img_shape[2]

    model = Sequential()
    model.add(Dense(image_height * image_width * 128, input_shape=(noise_dim,), kernel_initializer=weight_initializer))
    #model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM))
    #model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(Reshape((image_heigth, image_width, 128)))
    
    model = transposed_conv(model, 512, ksize=5, stride_size=1)
    model.add(Dropout(dropout_rate))
    model = transposed_conv(model, 256, ksize=5, stride_size=2)
    model.add(Dropout(dropout_rate))
    model = transposed_conv(model, 128, ksize=5, stride_size=2)
    model = transposed_conv(model, 64, ksize=5, stride_size=2)
    model = transposed_conv(model, 32, ksize=5, stride_size=2)
    
    model.add(Dense(3, activation='tanh', kernel_initializer=weight_initializer))

    #model.summary()

    noise = Input( shape=(noise_dim,) )
    img = model(noise)

    return Model(noise, img)

def build_discriminator(img_shape):

    image_height = img_shape[0]
    image_width = img_shape[1]
    image_channels = img_shape[2]

    model = Sequential()
    model.add(ConvSN2D(64, (5, 5), strides=(1,1), padding='same', use_bias=False, input_shape=[image_height, image_width, image_channels], kernel_initializer=weight_initializer))
    #model.add(BatchNormalization(epsilon=BN_EPSILON, momentum=BN_MOMENTUM))
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    #model.add(Dropout(dropout_rate))
    
    model = convSN(model, 64, ksize=5, stride_size=2)
    #model = convSN(model, 128, ksize=3, stride_size=1)
    model = convSN(model, 128, ksize=5, stride_size=2)
    #model = convSN(model, 256, ksize=3, stride_size=1)
    model = convSN(model, 256, ksize=5, stride_size=2)
    #model = convSN(model, 512, ksize=3, stride_size=1)
    #model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(DenseSN(1, activation='sigmoid'))

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