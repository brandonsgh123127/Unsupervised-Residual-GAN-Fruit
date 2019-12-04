import os

import cv2
import numpy as np
import glob
from Source.DatasetRetrieval import DatasetRetrieval


def main():
    #ImageML.start()
    dataset = DatasetRetrieval()
    data = dataset.retrieveImages()
    dir = "C:\\Users\\spada\\Desktop\\Samples\\"
    for filename in glob.glob(dir+"/*.png"):
        img = cv2.imread(filename)
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename,img)


main()