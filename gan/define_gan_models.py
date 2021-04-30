import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, \
                                    LeakyReLU, BatchNormalization,\
                                    Activation, UpSampling2D, \
                                    Conv2DTranspose, Dropout,\
                                    Conv2D, ReLU
from tensorflow.keras.models import Sequential, Model


NOISE_DIM = 128



def build_generator(img_shape):

    model = Sequential()

    model.add(Dense(7*7*256, input_dim=NOISE_DIM, use_bias=False))
    model.add(BatchNormalization(momentum=0.8))
    model.add(ReLU())

    model.add(Reshape((7,7,256)))
 
    model.add(UpSampling2D())
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(ReLU())
    
    model.add(UpSampling2D())
    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(ReLU())
    
    model.add(UpSampling2D())
    model.add(Conv2D(filters=img_shape[2], kernel_size=(5,5), strides=(1,1), padding="same", activation="tanh"))
    return model



def build_discriminator(img_shape):
    model = Sequential()
    
    model.add(Conv2D(filters=128,
                     kernel_size=(5,5),
                     strides=(1,1),
                     padding="same",
                     input_shape=img_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64, 
                     kernel_size=(5,5),
                     strides=(2,2),
                     padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     strides=(2,2),
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(units=1,
                    activation="sigmoid"))
    return model