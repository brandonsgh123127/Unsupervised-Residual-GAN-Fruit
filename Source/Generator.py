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
ops.reset_default_graph()






def generateFakeSamples(g_model, samples, patch_shape):
    print(samples.shape)
    X = g_model.predict(samples)
    print("len x", len(X), "patch shape ", patch_shape)
    Y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X,Y



"""
Point of this class is to create multiple generators for different types of fruit
Initial Size- 33 x 33
Target Size- 96 x 96
"""
class Generator(object):
    def __init__(self,image_shape=(32,32,3)):
        self.image_shape = image_shape

        #m_1 = Input(shape=(12,12,3))
        #m_1= Dense(10,input_dim=12)
    # A CONVOLUTION ALLOWS FOR 'HIDDEN LAYERS' TO BE PROCESSED
    # HIDDEN LAYERS ALLOW FOR CLASSIFICATION- PRODUCTION OF IMAGE
    """
    Create a convolution just for classifying...
    Next convolution is to build object based off last convolution...
    """
    def define_gen(self):
        init = RandomNormal(stddev=0.02)
        g_init = RandomNormal(mean=1.0,stddev=0.2)
        in_image = Input(shape=self.image_shape)
        print(self.image_shape," before generate")
        # ENCODE PROCESS FOR GENERATOR

        g0 = Conv2D(8,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(in_image)
        g0 = LeakyReLU(alpha=0.2)(g0)
        tmp = g0

        # B residual blocks  --  FAST FORWARDING to DEEPER LAYER
        for i in range(16):
            g1 = Conv2D(8, (3, 3), (1, 1), padding='same', kernel_initializer=init, bias_initializer=None)(g0)
            g1 = BatchNormalization(gamma_initializer=g_init)(g1)
            g1 = Activation('relu')(g1)
            g1 = Conv2D(8, (3, 3), (1, 1), padding='same', kernel_initializer=init, bias_initializer=None)(g1)
            g1 = BatchNormalization(gamma_initializer=g_init)(g1)
            g1 = Concatenate()([g0, g1])
            g0 = g1
        # 8 x 8 x 8
        g0 = Conv2D(8,(3,3),strides=(1,1),padding='same',kernel_initializer=init)(g0)
        g0 = BatchNormalization()(g0, training=True)
        g0 = Dropout(0.5)(g0, training=True)
        g0 = Concatenate()([g0,tmp])


        g0 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g0)
        g0 = UpSampling2D(size=(2, 2))(g0)
        g0 = Activation('relu')(g0)


        g0 = Conv2D(32, (3,3), strides=(1, 1), padding='same', kernel_initializer=init)(g0)
        g0 = UpSampling2D(size=(3, 3))(g0)
        g0 = Activation('relu')(g0)

        # changes channels
        g1= Conv2D(3, (3, 3), strides=(1, 1), padding='same', kernel_initializer=init)(g0)
        out_image = Activation('tanh')(g1)
        print(out_image.shape," boom")

        model = Model(in_image,out_image)
        return model
