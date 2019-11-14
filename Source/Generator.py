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
        self.dataset = []
        #m_1 = Input(shape=(12,12,3))
        #m_1= Dense(10,input_dim=12)
    # A CONVOLUTION ALLOWS FOR 'HIDDEN LAYERS' TO BE PROCESSED
    # HIDDEN LAYERS ALLOW FOR CLASSIFICATION- PRODUCTION OF IMAGE
    def generateFakeSamples(self,g_model,samples,patch_shape):
        X = g_model.predict(samples)
        Y = np.zeros((len(X),patch_shape,patch_shape,1))
        return X,Y
    """
    Create a convolution just for classifying...
    Next convolution is to build object based off last convolution...
    """
    def encoder(self,layer_in,n_filters,batchnorm=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2D(n_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        g = LeakyReLU(alpha=0.2)(g)
        return g

    def decoder(self,layer_in,skip_in,n_filters,dropout=True):
        init = RandomNormal(stddev=0.02)
        g = Conv2DTranspose(n_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)
        g = BatchNormalization()(g,training=True)
        if dropout:
            g = Dropout(.5)(g,training=True)
        g=Concatenate()([g,skip_in])
        g=Activation('relu')(g)
        return g

    def define_gen(self,imageShape=(256,256,3)):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=imageShape)
        e1 = self.encoder(in_image,64,batchnorm=False)
        e2=self.encoder(e1,128)
        e3=self.encoder(e2,256)
        e4=self.encoder(e3,512)
        e5=self.encoder(e4,512)
        e6=self.encoder(e5,512)
        e7=self.encoder(e6,512)

        b = Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(e7)
        b = Activation('relu')(b)
        d1 = self.decoder(b,e7,512)
        d2 = self.decoder(d1,e6,512)
        d3 = self.decoder(d2,e5,512)
        d4 = self.decoder(d3,e4,512,dropout=False)
        d5 = self.decoder(d4,e3,256,dropout=False)
        d6 = self.decoder(d5,e2,128,dropout=False)
        d7 = self.decoder(d6,e1,64,dropout=False)

        g = Conv2DTranspose(3,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d7)
        out_image = Activation('tanh')(g)

        model = Model(in_image,out_image)
        return model

    def define_discriminator(self,image_shape):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=image_shape)
        # target image input
        in_target_image = Input(shape=image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
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

    def train(self,d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
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
            [X_realA, X_realB], y_real = self.generateRealSamples(dataset, n_batch, n_patch)
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

    def generate_real_samples(self,dataset, n_samples, patch_shape):
        # unpack dataset
        trainA, trainB = dataset
        # choose random instances
        ix = randint(0, trainA.shape[0], n_samples)
        # retrieve selected images
        X1, X2 = trainA[ix], trainB[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y
    # train pix2pix model
    def train(self,d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
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
            [X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
            # generate a batch of fake samples
            X_fakeB, y_fake = self.generateFakeSamples(g_model, X_realA, n_patch)
            # update discriminator for real samples
            d_model.compile(optimizer=Adam(lr=0.0002,beta_1=0.5), loss='same')
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            # update discriminator for generated samples
            d_model.compile(optimizer=Adam(lr=0.0002,beta_1=0.5), loss='same')
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
            # summarize performance
            print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
            # summarize model performance
            if (i + 1) % (bat_per_epo * 10) == 0:
                self.summarize_performance(i, g_model, dataset)


    # generate samples and save as a plot and save the model
    def summarize_performance(self,step, g_model, dataset, n_samples=3):
        # select a sample of input images
        [X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake samples
        X_fakeB, _ = self.generateFakeSamples(g_model, X_realA, 1)
        # scale all pixels from [-1,1] to [0,1]
        X_realA = (X_realA + 1) / 2.0
        X_realB = (X_realB + 1) / 2.0
        X_fakeB = (X_fakeB + 1) / 2.0
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
    def start(self):
        # load image data
        dataset = DataR.DatasetRetrieval()
        self.dataset = dataset.retrieveImages()
        print('Loaded', self.dataset[0].shape, self.dataset[0].shape[1:])
        # define input shape based on the loaded dataset
        image_shape = self.dataset[0].shape[1:]
        # define the models
        d_model = self.define_discriminator(image_shape=(256,256,3))
        #d_model = self.define_gen(image_shape)
        g_model = self.define_gen()
        # define the composite model

        """
        BREAK POINT<<<<<  BROKEN!
        """
        gan_model = self.define_GAN(g_model, d_model, image_shape)
        # train model
        self.train(d_model, g_model, gan_model, self.dataset)
