import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import skimage
import os
import cv2 #USED FOR EDGE DETECTION IN IMAGES
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np

class Generator:
    NUM_ROWS = 28
    NUM_COLS = 28
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    EPOCHS = 10
    def __init__(self,PixArray,IMArray):
        print("To be implemented...")
