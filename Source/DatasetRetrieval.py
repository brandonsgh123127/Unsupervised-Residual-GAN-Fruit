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
        self.testSize = 3000
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


    ################################
    # Draw using matplotlib of 2 of same images
    # Will be used for comparisons....
    ################################
    def drawImageSample(self):
        self.clearFolder()  # Calls function that clears folder for new data

        #Sample data for user to see what machine looks at
        for item in range(0,5):
            """                 PLEASE CHANGE LINK TO LOCATION OF FRUIT                            """
            img = cv2.imread('E:\\Users\\i-pod\\Desktop\\Projects_CS\\Python\\Fruit-Images\\Fruit-Images-Dataset-master\\BasicFruit Images\\%s' % ' '.join(        #'E:\\Users\\i-pod\\Desktop\\Projects_CS\\Python\\Fruit-Images\\Fruit-Images-Dataset-master\\BasicFruit Images\\%s' 'C:\\Users\\spada\\OneDrive\\Documents\\CS368\\datasets\\BasicFruit Images\\%s
                map(str, self.imArray[item])))
            print(self.imArray[item])
            """
            Pixelate image given cv2's resize and interpolation...
            """
            width, height, _ = img.shape
            edges = cv2.Canny(img, width, height)  # CREATES AN EDGE DETECTION IMAGE
            w, h = (12, 12)  # New width/height of image...

            #Creates pixelated photos using Inter-Linear interpolation
            temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_BITS)
            output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_AREA)
            temp2 = cv2.Canny(temp, w, h)

            """
            # We need to save this data now to testLabeledData folder for use in Semi-Supervised Learning
            """
            cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\ORIG_%s' % ' '.join(
                map(str, self.imArray[item])),img)
            cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\EDGES_%s' % ' '.join(
                map(str, self.imArray[item])),edges)
            cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\PIXEDGE_%s' % ' '.join(
                map(str, self.imArray[item])),temp2)
            cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\PIX_%s' % ' '.join(
                map(str, self.imArray[item])),output)

            """
            
            Matplotlib plot data for user to see
            
            """
            fig = plt.figure(figsize=(12,12))
            fig.add_subplot(2, 2, 1).set(xlabel='pixelated'),\
            plt.imshow(output, )
            fig.add_subplot(2, 2, 2).set(xlabel='original'), plt.imshow(img, )
            fig.add_subplot(2, 2, 3).set(xlabel='pix edge'), plt.imshow(temp2,cmap='gray' )
            fig.add_subplot(2, 2, 4).set(xlabel='orig edge'), plt.imshow(edges,cmap='gray' )
            plt.show()
            #cv2.waitKey(0)

    #clears folder where test data will go...
    def clearFolder(self):
        # Iterates through files in test labeled data
        for f in os.listdir(os.getcwd()[:-7]+'\\TestLabeledData\\'):
            file_path = os.path.join(os.getcwd()[:-7]+'\\TestLabeledData\\', f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    """
    Attempt to upscale an image from edge pixelation
    """
    def upScale(self):
        print("To be implemented...")