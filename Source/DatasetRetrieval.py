import matplotlib
import pandas as pd
import sklearn
import skimage
import os
import cv2 #USED FOR EDGE DETECTION IN IMAGES

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
        self.imArray=np.zeros((self.testSize,1))
        self.imArray=self.imArray.astype(str)
        self.compArray=np.zeros(())

    """
    Used to retrieve a random set of images in dataset...
    """
    def retrieveImages(self):
        # Reads file named testPhotos, which points to each photo name
        with open(os.getcwd()[:-7]+'\\FileNames\\fruitNames.txt','r')as f:
            content = f.readlines()
            # Retrieve random number of objects in dataset to train model...
            for item in range(0,self.testSize):
                rand = np.random.randint(0,high=28590)
                self.imArray[item]= str(content[rand][:-1])  #imArray contains full quality image set locations
                #print(self.imArray[item])
                """                 PLEASE CHANGE LINK TO LOCATION OF FRUIT                            """
            img = cv2.imread('C:\\Users\\spada\\OneDrive\\Documents\\CS368\\datasets\\BasicFruit Images\\%s' % ' '.join(map(str,self.imArray[0])))
            print('C:\\Users\\spada\\OneDrive\\Documents\\CS368\\datasets\\BasicFruit Images\\%s' % ' '.join(map(str,self.imArray[0])))

            """
            Pixelate image given cv2's resize and interpolation...
            """
            width, height, _ = img.shape
            w, h = (16, 16)
            temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

            cv2.imshow('image',output)
            cv2.waitKey(0)

    #Using Kirsch Compass kernel
    def convImageEdges(self):
        print("To be implemented")
