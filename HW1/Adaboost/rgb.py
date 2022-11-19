from feature import RectangleRegion, HaarFeature
from classifier import WeakClassifier
import utils
import numpy as np
import math
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle
import os
import cv2 as cv
near_threshold = 0.1


def rgb_test(image_name,x,y,w,h):
    for i in os.walk('result/'):
        if image_name in i[2]:
            path = 'result/'
            break
        else:
            path = 'data/test/images/'

    imgo = cv.imread(path+image_name)
    # print(imgo.shape)
    wo=imgo.shape[1]
    ho=imgo.shape[0]
    img=cv.resize(cv.imread(path+image_name), (416, 416))
    img=img[max(int(y) - int(h / 2), 0):int(y) + int(h / 2),
          max(int(x) - int(w / 2), 0):int(x) + int(w / 2)]
    #cv.imshow("mask", img)
    #cv.waitKey(0)
    
    imgo = cv.resize(imgo,(416,416))
    if classify(img)==0:
        cv.rectangle(imgo, (max(int(x) - int(w / 2), 0), max(int(y) - int(h / 2), 0)), (int(x) + int(w / 2), int(y)+int(h/ 2)), (0, 255, 0), 2)
        #cv.imshow("mask", imgo)
        #cv.waitKey(0)
    else:
        cv.rectangle(imgo, (max(int(x) - int(w / 2), 0), max(int(y) - int(h / 2), 0)), (int(x) + int(w / 2), int(y)+int(h/ 2)), (0, 0, 255), 2)
    cv.imwrite('result/'+image_name, cv.resize(imgo,(wo,ho)) )
    
def classify(image):
        d=image
        y1,x2,col=image.shape
        
        x1 = 0
        y2 = 0
        
        faceLong = y1-y2
        #print(y1,y2)
        downface = image[int(1/3*faceLong):y1,x1:x2]
        upface = image[y2:int(1/3*faceLong),x1:x2]
        #cv.imshow("Image",downface)
        #cv.waitKey (0) 
        hist1 = cv.calcHist([upface], [0,1,2], None, [256,256,256], [0, 256, 0, 256,0, 256])
        hist2 = cv.calcHist([downface], [0,1,2], None, [256,256,256], [0, 256, 0, 256,0, 256])
        #cv.imshow("Image",upface)
        #cv.waitKey (0) 
        # 平移縮放
        cv.normalize(hist1, hist1, 0, 1.0, cv.NORM_MINMAX)
        cv.normalize(hist2, hist2, 0, 1.0, cv.NORM_MINMAX)
        
        near = cv.compareHist(hist1,hist2,0)
        print(near)
        return 1 if near > near_threshold else 0
