import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2 #USED FOR EDGE DETECTION IN IMAGES
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense

"""
Point of this class is to create multiple generators for different types of fruit
Initial Size- 12 x 12
Target Size- 100 x 100
"""
class Generator:
    def __init__(self,PixArray,IMArray):
        self.NUM_ROWS = 28
        self.NUM_COLS = 28
        self.NUM_CLASSES = 10
        self.BATCH_SIZE = 128
        self.EPOCHS = 5
        m_1 = Input
    # A CONVOLUTION ALLOWS FOR 'HIDDEN LAYERS' TO BE PROCESSED
    # HIDDEN LAYERS ALLOW FOR CLASSIFICATION- PRODUCTION OF IMAGE
    """
    Create a convolution just for classifying...
    Next convolution is to build object based off last convolution...
    """
