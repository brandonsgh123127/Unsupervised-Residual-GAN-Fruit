import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array
import Generator as generator
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
        self.testSize = 1000 # CHANGE LATER
        self.compArray=np.zeros(())
        self.pixArray = np.zeros(()) # STORES 12x12 IMAGES
        self.imArray=np.zeros((self.testSize,1))
        self.imArray=self.imArray.astype(str) # STORES LOCATIONS TO IMAGES
        self.edgeArray= np.zeros(()) # STORES 100 x 100 IMAGES
        self.edgeArray=self.edgeArray.astype(np.ndarray)
        self.edgePixArray= np.zeros(()) # STORES 12x12 IMAGES
        self.edgePixArray=self.edgePixArray.astype(np.ndarray)
        self.origArray = np.zeros(()) # STORES 100 x 100 IMAGES
        self.origArray=self.origArray.astype(np.ndarray)
        self.src_images=[]
        self.tar_images=[]

    """
    Used to retrieve a random set of images in dataset...
    """
    def retrieveImages(self):
        # Reads file named testPhotos, which points to each photo name
        with open(os.getcwd()+'\\FileNames\\fruitNames.txt','r',encoding='utf8',newline='\r\n')as f:
            content = f.readlines()
            # Retrieve random number of objects in dataset to train model...
            for item in range(0,self.testSize):
                rand = np.random.randint(0,high=28587)
                self.imArray[item]= ''.join(str(content[rand][0:-5]).strip('\r\n') + 'jpg')
                print(self.imArray[item],"fdjkjk")
                #self.imArray[item]= str(content[rand].strip('\\'))  #imArray contains full quality image set locations
            return self.drawImageSample(self.testSize,'E:\\Users\\i-pod\\Desktop\\Projects_CS\\Python\\Semi-Supervised-Learning\\BasicFruit Images\\BasicFruit Images')


    ################################
    # Draw using matplotlib of 2 of same images
    # Will be used for comparisons....
    ################################
    def drawImageSample(self,sample_size,location):
        self.clearFolder()  # Calls function that clears folder for new data
        src_list, tar_list = list(), list()
        #Sample data for user to see what machine looks at
        for item in range(0,sample_size):

            """                 PLEASE CHANGE LINK TO LOCATION OF FRUIT                            """
            img = cv2.imread(location + '\\%s' % ' '.join(self.imArray[item]))
            print(self.imArray[item])
            """
            Pixelate image given cv2's resize and interpolation...
            """
            """
            Just in case string gets cut off of file location, try to add 'g' to end
            """
            try:
                width, height, _ = img.shape
            except:
                tmp = str(self.imArray[item])
                tmp = tmp.strip('[]').strip('\'').rstrip('g')
                tmp+='g'
                print(tmp)
                img = cv2.imread(location + '\\%s' % ''.join(
                    tmp))
                width, height, _ = img.shape


            #edges = cv2.Canny(img, width, height)  # CREATES AN EDGE DETECTION IMAGE
            w, h = (32, 32)  # New width/height of image...
            #Creates pixelated photos using Inter-Linear interpolation
            temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_BITS)
            #output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_AREA)
            #temp2 = cv2.Canny(temp, w, h)

           # cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\ORIG_%s' % ' '.join(
            #    map(str, self.imArray[item])),img)
            #cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\EDGES_%s' % ' '.join(
            #    map(str, self.imArray[item])),edges)
            #cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\PIXEDGE_%s' % ' '.join(
            #    map(str, self.imArray[item])),temp2)
            #cv2.imwrite(os.getcwd()[:-7]+'\\TestLabeledData\\PIX_%s' % ' '.join(
           #     map(str, self.imArray[item])),output)
            """
            Matplotlib plot data for user to see
            """
            #fig = plt.figure(figsize=(12,12))
            #fig.add_subplot(2, 2, 1).set(xlabel='pixelated'),\
            #plt.imshow(output, )
            self.addPixArray(temp)
            # fig.add_subplot(2, 2, 2).set(xlabel='original'), plt.imshow(img, )
            self.addOrigArray(img)
            self.origData = img_to_array(img)
            self.pixData = img_to_array(temp)
            #fig.add_subplot(2, 2, 3).set(xlabel='pix edge'), plt.imshow(temp2,cmap='gray' )
            #self.addEdgePixArray(temp2)
            #fig.add_subplot(2, 2, 4).set(xlabel='orig edge'), plt.imshow(edges,cmap='gray' )
            #self.addEdgeArray(edges)
            orig_img, pix_img = self.origData[:, :sample_size], self.pixData[:, :sample_size]
            src_list.append(orig_img)
            tar_list.append(pix_img)
            #plt.show()
            #cv2.waitKey(0)
            #print(self.edgeArray.size)
        """
        In order to use for 2 variables, save data to numpy file
        """
        self.src_images,self.tar_images = np.asarray(src_list), np.asarray(tar_list)
        np.savez_compressed("fruits.npz",self.src_images,self.tar_images)
        print("Saved to 'fruits.npz'!!")
        self.src_images,self.tar_images= self.loadData("fruits.npz")
        print(self.src_images[50][50][50][1])
        return np.asarray(self.src_images),np.asarray(self.tar_images)#,np.ones((sample_size, 1, 1, 1))


    # clears folder where test data will go...
    def clearFolder(self):
        # Iterates through files in test labeled data
        for f in os.listdir(os.getcwd()+'\\TestLabeledData\\'):
            file_path = os.path.join(os.getcwd()+'\\TestLabeledData\\', f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    """
    Getters and setters for above arrays...
    """
    def getIMArray(self):
        return self.imArray
    def getOrigArray(self):
        return self.origArray
    def getPixArray(self):
        return self.pixArray
    def getEdgePixArray(self):
        return self.edgePixArray
    def getEdgeArray(self):
        return self.edgeArray
    def addOrigArray(self,image):
        np.append(self.origArray,image)
        print("Added original photo")
    def addPixArray(self,image):
        np.append(self.pixArray,image)
        print("Added pixel photo")
    def addEdgePixArray(self,image):
        np.append(self.edgePixArray,image)
        print("Added edge pixel photo")
    def addEdgeArray(self,image):
        np.append(self.edgeArray,image)
        print("Added edge photo")



    def loadData(self,filename):
        data = np.load(filename)
        self.src_images, self.tar_images = data['arr_0'], data['arr_1']
        print('Loaded: ', self.src_images.shape, self.tar_images.shape)
        n_samples = 3
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(self.src_images[i].astype('uint8'))
        # plot target image
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(self.tar_images[i].astype('uint8'))
        plt.show()
        self.src_images=normalize(self.src_images)
        self.tar_images=normalize(self.tar_images)
        return [self.src_images,self.tar_images]


def normalize(src):
    src = (src - 127.5) / 127.5
    return src


def denormalize(src):
    src = (src +1) * 127.5
    return src

    """ 
"""