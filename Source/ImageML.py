import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2 #USED FOR EDGE DETECTION IN IMAGES
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow_core.python.ops import nn

import DatasetRetrieval as DataR
from tensorflow.keras.applications.vgg19 import VGG19


from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend
from tensorflow.python.framework import ops

import Discriminator
import Generator

ops.reset_default_graph()


"""
# Defines a Generative Adversarial Network(GAN), in which an image will get upscaled from a 32x32x3
# to a 96x96x3 through Residual Network.  Dataset: https://drive.google.com/open?id=12EmrkOb5wR7-o33EZ4dmnPLsqhGpfXLM
# Dataset is originally from: 
#
"""
def define_GAN(g_model,d_model,image_shape):
    d_model.trainable = False
    in_src = Input(shape=(32,32,3)) # 32 32 3
    out_src = Input(shape=(96,96,3))
    gen_out = g_model(in_src)
    dis_out = d_model(gen_out)
    model = Model(inputs=in_src,outputs=[gen_out,dis_out])
    #model = Model(in_src,[gen_out,dis_out])
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "binary_crossentropy"],optimizer=opt)
    #generator_model = Model(inputs=in_src, outputs=model)
    return model

"""
Retrieves separate images to be used for GAN Network,
X1, is original hr image, X2 is original lr image
"""
def generate_real_samples(dataset, n_samples, n_patch):
    # Dataset pushed to a and b
    # WHERE A IS REAL ORIG IMAGE AND B IS PIXELATED IMAGE
    train_hr, train_lr = dataset

    # Random value of dataset
    rand = randint(0, train_hr.shape[0], n_samples)

    X1, X2 = train_hr[rand], train_lr[rand]

    # generate 'real' class labels (1)
    y1 = np.ones((n_samples, n_patch, n_patch, 1))
    y2 = np.ones((n_samples, 96, 96, 1))
    return X1, X2, y1

"""
# TO BE IMPLEMENTED!!!!!!!
"""
def build_vgg(gen):
    vgg = VGG19(weights="imagenet")
    # Set outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[9].output]
    img = Input(shape=(96,96,3))
    # Extract image features
    img_features = vgg(img)
    return Model(img, img_features)
"""
# After an epoch, normalize data, plot it on a chart to show data, then save to h5 file.
#
"""
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    X_real_hr, X_real_lr, y_real = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fake_hr, _ = Generator.generateFakeSamples(g_model, X_real_lr, 1)

    # scale all pixels from [-1,1] to [0,1]
    X_realA = ((X_real_hr + 1) / 2.0)
    X_realB = ((X_real_lr + 1) / 2.0)
    X_fakeB = ((X_fake_hr + 1) / 2.0)
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

"""
# Start function allows for GAN Network to be created.  Creates Discriminator,
# Generator, and GAN Network which combines both.
"""
def start():
    # THIS IS THE DATA THAT IS RETRIEVED IN DATASET
    dataset = DataR.DatasetRetrieval()
    dataset = dataset.retrieveImages() # first half is orig images, 2nd half is pixelated images
    print('Loaded ', dataset[0].shape, dataset[0].shape[1:], " Image sizes")
    # Image shape is 96 x 96 x 3 in this dataset
    image_shape = dataset[0].shape[1:]

    # define descriminator model
    descrim_model = Discriminator.Discriminator(image_shape)
    descrim_model= descrim_model.define_discriminator()

    # Define generator model
    gen_model = Generator.Generator((32,32,3))
    gen_model= gen_model.define_gen()

    # GAN MODEL IMPLEMENTS BOTH GENERATOR AND DESCRIMINATOR INSIDE
    gan_model = define_GAN(gen_model, descrim_model,image_shape)
    n_patch = descrim_model.get_output_shape_at(1)[1] # size 1

    n_batches= dataset[0].shape[0]
    # unpack dataset
    train_hr, train_lr = dataset
    # num of batches per epoch
    ####################################
    #
    # Train Discriminator...
    #
    #####################################
    bat_per_epo = int(len(train_hr) / n_batches)
    # Calculates total iterations needed based on epochs (100 epochs)
    n_steps = bat_per_epo * 100
    # iterate through each epoch through steps
    for i in range(n_steps):
        # retrieve real samples
        X_real_hr, X_real_lr,real_y1= generate_real_samples(dataset, n_batches, n_patch)
        # generate fake images
        X_fakeB,fake_y = Generator.generateFakeSamples(gen_model, X_real_lr, n_patch,)

        #X_real_hr = (X_real_hr + 1) / 2.0
        #X_real_lr = (X_real_lr + 1) / 2.0
        #X_fakeB = (X_fakeB + 1) / 2.0
        # Loss function of first set of real images
        _,d_loss_real = descrim_model.train_on_batch(X_real_hr,real_y1)

        # Loss function for fake images
        _,d_loss_fake = descrim_model.train_on_batch(X_fakeB,fake_y)
        d_loss= 0.5 * np.add(d_loss_real,d_loss_fake)# d_loss[0]--> shape(2,)
        _,g_loss,_= gan_model.train_on_batch(X_real_lr, [X_real_hr,real_y1]) #in_src,[gen_out,dis_out] model

        # Loss functions printed out
        print('>%d, dreal[%.4f], dfake[%.4f], g[%.4f]' % (i + 1, d_loss_real, d_loss_fake,g_loss))
        # save data after epoch
        if (i + 1) % (5) == 0:
            summarize_performance(i, gen_model, dataset)
