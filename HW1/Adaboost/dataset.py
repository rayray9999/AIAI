import os
import cv2
import numpy as np
def loadImages(dataPath):
    ag1=[] 
    lb1=[]
    # this function is for read image,the input is directory name
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(dataPath+"/withMask"):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(dataPath + "/withMask/" + filename)
        ag1.append(img)
        lb1.append(1)

    for filename in os.listdir(dataPath+"/withoutMask"):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(dataPath + "/withoutMask/" + filename)
        ag1.append(img)
        lb1.append(0)
        
    #raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    dataset=tuple(zip(ag1,lb1))
    return dataset
