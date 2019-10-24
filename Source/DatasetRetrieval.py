import matplotlib
import pandas as pd
import sklearn
import skimage

import numpy as np


"""
# The purpose of this class is to get a preprocessed data set in order for Neural Network to understand
# 
# what to do.  Since this project is classifying multiple unlabeled sets, it is important to train first.
#
"""
class DatasetRetrieval:
    global imArray #Used to store images inside

    # Used to initialize the class/Object when created
    def __init__(self):
        self.testSize = 250
        self.imArray= np.zeros((1,self.testSize))
    """
    Used to retrieve a random set of images in dataset...
    """
    def retrieveImages(self):
        # Reads file named testPhotos, which points to each photo name
        with open('trainPhotos.txt','w')as f:
            # Retrieve random number of objects in dataset to train model...
            for item in range(0,self.testSize):
                print("To be implemented")
                f.write("s")


start = DatasetRetrieval()
start.retrieveImages()