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
from tensorlayer.layers import Reshape

import Source.DatasetRetrieval as DataR
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

from Source import Discriminator, Generator

ops.reset_default_graph()



def define_GAN(g_model,d_model):
    d_model.trainable = False
    in_src = Input(shape=(32,32,3))
    out_src = Input(shape=(96,96,3))
    gen_out = g_model(in_src)
    dis_out = d_model([out_src,gen_out])
    model = Model([in_src,out_src],[dis_out])
    #model = Model(in_src,[gen_out,dis_out])
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss=['binary_crossentropy'],optimizer=opt)
    #generator_model = Model(inputs=in_src, outputs=model)
    return model


def generate_real_samples(dataset, n_samples, patch_shape):
    # Dataset pushed to a and b
    # WHERE A IS REAL ORIG IMAGE AND B IS PIXELATED IMAGE
    trainA, trainB = dataset

    # Random value of dataset
    rand = randint(0, trainA.shape[0], n_samples)

    X1, X2 = trainA[rand], trainB[rand]

    # generate 'real' class labels (1)
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return X1, X2, y


def build_vgg(X_real_hr):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg = VGG19(weights="imagenet")
    # Set outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[9].output]

    img = Input(shape=X_real_hr.shape)

    # Extract image features
    img_features = vgg(img)

    return Model(img, img_features)
# Save each set and plot
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    X_real_hr, X_real_lr, y_real = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    image_features = vgg.predict()

    X_fake_hr, _ = Generator.generateFakeSamples(g_model, X_real_lr, 1)

    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_real_hr + 1) / 2.0
    X_realB = (X_real_lr + 1) / 2.0
    X_fakeB = (X_fake_hr + 1) / 2.0
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

    vgg = build_vgg()
    vgg.trainable = False
    vgg.compile(loss='mse',
                optimizer=Adam(0.0002,0.5),
                metrics=['accuracy'])
    # GAN MODEL IMPLEMENTS BOTH GENERATOR AND DESCRIMINATOR INSIDE
    gan_model = define_GAN(gen_model, descrim_model)
    #
    #self.train(descrim_model, gen_model, gan_model, self.dataset)

    print(descrim_model.get_output_shape_at(1))
    n_patch = descrim_model.get_output_shape_at(0)[1]
    n_batches=1
    # unpack dataset
    train_hr, train_lr = dataset
    # num of batches per epoch


    ####################################
    #
    # Train Discriminator...
    #
    #####################################
    patch = int((96 / 2 ** 4))
    disc_patch = (patch, patch, 1)
    valid = np.ones(disc_patch)
    fake = np.zeros((disc_patch))
    bat_per_epo = int(len(train_hr) / n_batches)
    # Calculates total iterations needed based on epochs (100 epochs)
    n_steps = bat_per_epo * 100
    # manually enumerate epochs
    for i in range(n_steps):
        # get real samples
        X_real_hr, X_real_lr,real_y = generate_real_samples(dataset, n_batches, n_patch)
        # generate fakes
        X_fakeB,fake_y = Generator.generateFakeSamples(gen_model, X_real_lr, n_patch,)

        X_real_hr = (X_real_hr + 1) / 2.0
        X_real_lr = (X_real_lr + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0

        fake_features = vgg(X_fakeB)

        # Loss function of first set of real images
        d_loss_real = descrim_model.train_on_batch(X_real_hr,real_y)

        # Loss function for fake images
        d_loss_fake = descrim_model.train_on_batch(X_fakeB,fake_y)


        d_loss= 0.5 * np.add(d_loss_real,d_loss_fake)


        g_loss= gan_model.train_on_batch([X_real_lr, X_real_hr], [real_y, vgg])

        # Loss functions of descriminator and generators
        #print('>%d, d2[%.4f] g[%.4f]' % (i + 1, d_loss_real, d_loss_fake, d_loss))

        # every end of epoch, save data.....
        if (i + 1) % (100) == 0:
            summarize_performance(i, gen_model, dataset)
