import matplotlib.pyplot as plt
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
            self.drawImageSample()


    #Using Kirsch Compass kernel
    def convImageEdges(self):
        print("To be implemented")

    ################################
    # Draw using matplotlib of 2 of same images
    # Will be used for comparisons....
    ################################
    def drawImageSample(self):
        #Sample data for user to see what machine looks at
        for item in range(0,5):
            """                 PLEASE CHANGE LINK TO LOCATION OF FRUIT                            """
            img = cv2.imread('C:\\Users\\spada\\OneDrive\\Documents\\CS368\\datasets\\BasicFruit Images\\%s' % ' '.join(
                map(str, self.imArray[item])))
            print(self.imArray[item])
            """
            Pixelate image given cv2's resize and interpolation...
            """
            width, height, _ = img.shape
            edges = cv2.Canny(img, width, height)  # CREATES AN EDGE DETECTION IMAGE
            w, h = (16, 16)  # New width/height of image...
            temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            temp2 = cv2.Canny(temp, w, h)
            output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

            # cv2.imshow('image',output)
            fig = plt.figure(figsize=(12,12))
            fig.add_subplot(2, 2, 1).set(xlabel='pixelated'),\
            plt.imshow(output, )
            fig.add_subplot(2, 2, 2).set(xlabel='original'), plt.imshow(img, )
            fig.add_subplot(2, 2, 3).set(xlabel='pix edge'), plt.imshow(temp2,cmap='gray' )
            fig.add_subplot(2, 2, 4).set(xlabel='orig edge'), plt.imshow(edges,cmap='gray' )
            plt.show()
            #cv2.waitKey(0)

    """
    Attempt to upscale an image from edge pixelation
    """
    def upScale(self):
        print("To be implemented...")