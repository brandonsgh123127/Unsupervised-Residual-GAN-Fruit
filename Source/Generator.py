import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2 #USED FOR EDGE DETECTION IN IMAGES
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import Source.DatasetRetrieval as DataR


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
Target Size- 96 x 96
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


    def define_discriminator(self,image_shape):
        # initialize weight
        initial = RandomNormal(stddev=0.02)
        # source image
        in_src_image = Input(shape=image_shape)
        # target image
        in_target_image = Input(shape=image_shape)
        # place both source and target images in one
        merged_images = Concatenate()([in_src_image, in_target_image])
        """
        CONVOLUTIONAL NETWORK FOR DISCRIMINATOR!!!   
        """
        d = Conv2D(32,(4,4),strides=(2,2),padding='same',kernel_initializer=RandomNormal(stddev=0.02))(merged_images)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(96,(4,4),strides=(2,2),padding='same',kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)


        d = Conv2D(192,(4,4),strides=(2,2),padding='same',kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(192,(4,4),strides=(2,2),padding='same',kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(axis=-1)(d)
        d = LeakyReLU(alpha=0.2)(d)


        d = Conv2D(1,(4,4),padding='same',kernel_initializer=RandomNormal(stddev=0.02))(d)
        patch_out = Activation('sigmoid')(d)
        # define model
        model = Model([in_src_image, in_target_image], patch_out)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model



    def define_gen(self,image_shape=(96,96,3)):
        init = RandomNormal(stddev=0.02)
        in_image = Input(shape=image_shape)
        # ENCODE PROCESS FOR GENERATOR
        g1 = Conv2D(32,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(in_image)
        g1 = LeakyReLU(alpha=0.2)(g1)

        g2 = Conv2D(96,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(g1)
        g2 = BatchNormalization()(g2,training=True)
        g2 = LeakyReLU(alpha=0.2)(g2)

        g3 = Conv2D(192, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g2)
        g3 = LeakyReLU(alpha=0.2)(g3)
        g3 = BatchNormalization()(g3,training=True)

        g4 = Conv2D(192, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g3)
        g4 = LeakyReLU(alpha=0.2)(g4)
        g4 = BatchNormalization()(g4, training=True)



        # BOTTLE NECK THE ENCODER
        b = Conv2D(192, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g4)
        b = Activation('relu')(b)

        # DECODE THE ENCODED NETWORK
        g5 = Conv2DTranspose(192, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(b)
        g5 = BatchNormalization()(g5, training=True)
        g5 = Concatenate()([g5, g4])
        g5 = Dropout(0.5)(g5, training=True)
        g5 = Activation('relu')(g5)

        g6 = Conv2DTranspose(192, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g5)
        g6 = BatchNormalization()(g6, training=True)
        g6 = Concatenate()([g6, g3])
        g6 = Dropout(0.5)(g6, training=True)
        g6 = Activation('relu')(g6)

        g7 = Conv2DTranspose(96, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g6)
        g7 = BatchNormalization()(g7, training=True)
        g7 = Concatenate()([g7, g2])
        g7 = Activation('relu')(g7)

        g8 = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g7)
        g8 = BatchNormalization()(g8, training=True)
        g8 = Concatenate()([g8, g1])
        g8 = Activation('relu')(g8)


        g9= Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(g8)
        out_image = Activation('tanh')(g9)

        model = Model(in_image,out_image)
        return model

    def define_GAN(self,g_model,d_model,image_shape):
        d_model.trainable = False
        in_src = Input(shape=(96,96,3))
        gen_out = g_model(in_src)
        dis_out = d_model([in_src,gen_out])
        model = Model(in_src,[dis_out,gen_out])
        opt = Adam(lr=0.0002,beta_1=0.5)
        model.compile(loss=['binary_crossentropy','mae'],optimizer=opt,loss_weights=[1,100])
        return model


    def generate_real_samples(self,dataset, n_samples, patch_shape):
        # Dataset pushed to a and b
        # WHERE A IS REAL ORIG IMAGE AND B IS PIXELATED IMAGE
        trainA, trainB = dataset

        # Random value of dataset
        rand = randint(0, trainA.shape[0], n_samples)
        # retrieve images
        X1, X2 = trainA[rand], trainB[rand]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
        return [X1, X2], y


    # Save each set and plot
    def summarize_performance(self,step, g_model, dataset, n_samples=3):
        # select a sample of input images
        [X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
        # generate a batch of fake samples
        X_fakeB, _ = self.generateFakeSamples(g_model, X_realB, 1)
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

        # THIS IS THE DATA THAT IS RETRIEVED IN DATASET
        dataset = DataR.DatasetRetrieval()
        self.dataset = dataset.retrieveImages()


        print('Loaded ', self.dataset[0].shape, self.dataset[0].shape[1:], " Image sizes")
        # Image shape is 96 x 96 x 3 in this dataset
        image_shape = self.dataset[0].shape[1:]
        # define descriminator model
        descrim_model = self.define_discriminator(image_shape)
        #d_model = self.define_gen(image_shape)
        gen_model = self.define_gen()


        # GAN MODEL USES BOTH A GENERATOR AND DESCRIMINATOR INSIDE....
        gan_model = self.define_GAN(gen_model, descrim_model, image_shape)
        #
        #self.train(descrim_model, gen_model, gan_model, self.dataset)

        print(descrim_model.get_output_shape_at(1))
        n_patch = descrim_model.get_output_shape_at(1)[1]
        n_batches=1
        # unpack dataset
        trainA, trainB = self.dataset
        # num of batches per epoch
        bat_per_epo = int(len(trainA) / n_batches)
        # Calculates total iterations needed based on epochs (100 epochs)
        n_steps = bat_per_epo * 100
        # manually enumerate epochs
        for i in range(n_steps):
            # get real samples
            [X_realA, X_realB], y_real = self.generate_real_samples(self.dataset, n_batches, n_patch)
            # generate fakes
            X_fakeB, y_fake = self.generateFakeSamples(gen_model, X_realB, n_patch)


            # Loss function of first set of real images
            d_loss1 = descrim_model.train_on_batch([X_realA, X_realB], y_real)

            # Loss function for fake images
            d_loss2 = descrim_model.train_on_batch([X_realA, X_fakeB], y_fake)

            # update the generator
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

            # Loss functions of descriminator and generators
            print('>%d, d1[%.4f] d2[%.4f] g[%.4f]' % (i + 1, d_loss1, d_loss2, g_loss))

            # every end of epoch, save data.....
            if (i + 1) % (bat_per_epo) == 0:
                self.summarize_performance(i, gen_model, self.dataset)
