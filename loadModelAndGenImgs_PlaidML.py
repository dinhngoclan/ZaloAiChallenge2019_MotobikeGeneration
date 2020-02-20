# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:16:02 2019

@author: laptop
"""

# Only 2 lines will be added
# Rest of the flow and code remains the same as default keras
import plaidml.keras as keras
keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.install_backend"
import numpy
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras import backend as k 
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
from numpy import savez_compressed
 
# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 8x8 feature maps
    n_nodes = 128 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))
    # upsample to 10x10
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 20x20
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 40x40
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 128x128
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer 128x128x3
    model.add(Conv2D(3, (5,5), activation='tanh', padding='same'))
    return model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

# create a plot of generated images
def plot_generated(examples, n):
    print ('shape3 %s' % numpy.shape(examples)[3])
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n):
        # define subplot
        fig, ax = pyplot.subplots(nrows=1,ncols=1)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
        filename = './GeneratedImages/img%03d.png' % (i+1)
        fig.savefig(filename)
        pyplot.close(fig)
    return
 
# load model
# create the generator
latent_dim = 100
g_model = define_generator(latent_dim)
#Continue trainning from saved model   
g_model.load_weights('generator_model_039.h5')
#model = load('./generated_plot_e002.h5')
# generate points in latent space
latent_points = generate_latent_points(latent_dim, 10000)
# generate images
examples = g_model.predict(latent_points)

# save plotgenerate_latent_points
plot_generated(examples, 10000)
