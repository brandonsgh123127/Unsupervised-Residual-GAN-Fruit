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
ops.reset_default_graph()




class Discriminator(object):
    def __init__(self,image_shape):
        self.image_shape = image_shape
    def define_discriminator(self):
        # initialize weight
        # source image
        print(self.image_shape, "ffffff")
        g_init = RandomNormal(mean=1.0, stddev=0.2)
        init = RandomNormal(stddev=0.2)
        in_src_image = Input(shape=self.image_shape)
        print("descrim shape int ", self.image_shape)
        # target image
        #in_target_image = Input(shape=self.image_shape)
        # place both source and target images in one
       # merged_images = Concatenate()([in_src_image, in_target_image])
        """
        CONVOLUTIONAL NETWORK FOR DISCRIMINATOR!!!   
        """
        d = Conv2D(32, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(in_src_image)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(1024, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer=init)(d)
        d1 = BatchNormalization(gamma_initializer=g_init)(d)

        d = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d1)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=RandomNormal(stddev=0.02))(d)
        d = BatchNormalization(gamma_initializer=g_init)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Concatenate()([d, d1])
        d = LeakyReLU(alpha=0.2)(d)

        d = Flatten()(d)
        d = Dropout(0.2)(d)
        # d = Dense(1024,kernel_initializer=init)(d)
        # d = LeakyReLU(alpha=0.2)(d)

        # d = Conv2D(1, (3,3), strides = (1,1),padding = 'same')(d)

        d2 = Dense(1, kernel_initializer=init)(d)

        # d = Activation('sigmoid')(d)

        # define model
        model = Model(in_src_image,d2)
        # compile model
        # opt = Adam(lr=0.0002, beta_1=0.5)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        #discriminator_model = Model(inputs=in_src_image, outputs=model)
        return model


