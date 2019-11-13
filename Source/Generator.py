import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2 #USED FOR EDGE DETECTION IN IMAGES
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import DatasetRetrieval as DataR


from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend


from tensorflow.python.framework import ops
ops.reset_default_graph()


"""
Point of this class is to create multiple generators for different types of fruit
Initial Size- 33 x 33
Target Size- 100 x 100
"""
class Generator:
    def __init__(self):
        self.NUM_ROWS = 28
        self.NUM_COLS = 28
        self.NUM_CLASSES = 10
        self.BATCH_SIZE = 128
        self.EPOCHS = 5
        #m_1 = Input(shape=(12,12,3))
        #m_1= Dense(10,input_dim=12)
    # A CONVOLUTION ALLOWS FOR 'HIDDEN LAYERS' TO BE PROCESSED
    # HIDDEN LAYERS ALLOW FOR CLASSIFICATION- PRODUCTION OF IMAGE
    def generateFakeSamples(self,g_model,samples,patch_shape):
        X = g_model.predict(samples)
        Y = zeros((len(X),patch_shape,patch_shape,1))
        return X,Y
    """
    Create a convolution just for classifying...
    Next convolution is to build object based off last convolution...
    """
    def encoder(self,layer_in,n_filters,batchnorm=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2D(n_filters,(3,3),strides=(1,1),padding='same',kernel_initializer=init)(layer_in)
        g = BatchNormalization()(g,training=True)
        g = LeakyReLU(alpha=0.2)(g)
        return g

    def decoder(self,layer_in,skip_in,n_filters,dropout=True):
        init = RandomNormal(stddev=0.02)
        print(layer_in.shape)
        gg = Conv2DTranspose(n_filters,(4,4),strides=(3,3),padding='same',kernel_initializer=init)(layer_in)
        g = BatchNormalization()(gg,training=True)
        if dropout:
            g = Dropout(.5)(g,training=True)
        g=Concatenate()([g,skip_in])
        g=Activation('relu')(g)
        return g

    def define_gen(self,imageShape=(100,100,3)):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=imageShape)
        e1 = self.encoder(in_image,44,batchnorm=False)
        e2=self.encoder(e1,66)
        e3=self.encoder(e2,88)
        e4=self.encoder(e3,100)
        e5=self.encoder(e4,100)
        e6=self.encoder(e5,100)
        e7=self.encoder(e6,100)

        b = Conv2D(100,(3,3),strides=(4,4),padding='same',kernel_initializer=init)(e7)
        b = Activation('relu')(b)


        d0 = self.decoder(b,e7,100)
        d1 = self.decoder(d0,e6,100)
        d2 = self.decoder(d1,e5,100)
        d3 = self.decoder(d2,e4,100,dropout=False)
        d4 = self.decoder(d3,e3,88,dropout=False)
        d5 = self.decoder(d4,e2,66,dropout=False)
        d6 = self.decoder(d5,e1,44,dropout=False)

        g = Conv2DTranspose(3,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d6)
        out_image = Activation('tanh')(g)

        model = Model(in_image,out_image)
        return model
    def define_GAN(self,g_model,d_model,image_shape):
        d_model.trainable = False
        in_src = Input(shape=image_shape)
        gen_out = g_model(in_src)
        dis_out = d_model([in_src,gen_out])
        model = Model(in_src,[dis_out,gen_out])
        opt = Adam(lr=0.0002,beta_1=0.5)
        model.compile(loss=['binary_crossentropy','mae'],optimizer=opt,loss_weights=[1,100])
        return model

    def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
        # determine the output square shape of the discriminator
        n_patch = d_model.output_shape[1]
        # unpack dataset
        trainA, trainB = dataset
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(trainA) / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # manually enumerate epochs
        for i in range(n_steps):
            # select a batch of real samples
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
            # summarize model performance
            if (i + 1) % (bat_per_epo * 10) == 0:
                summarize_performance(i, g_model, dataset)

    def start(self):
        # load image data
        dataset = DataR.DatasetRetrieval()
        dataset = dataset.retrieveImages()
        print('Loaded', dataset[0].shape, dataset[0].shape[1:])
        # define input shape based on the loaded dataset
        image_shape = dataset[0].shape[1:]
        # define the models
        #d_model = discriminator(image_shape)
        #d_model = self.define_gen(image_shape)
        g_model = self.define_gen(image_shape)
        # define the composite model
        gan_model = self.define_gan(g_model, g_model, image_shape)
        # train model
        train(g_model, g_model, gan_model, dataset)