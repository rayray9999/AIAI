# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2 as cv
import numpy as np
f=open('bounding_box.txt','r')
#cv.startWindowThread()
S=f.read().split()
p=list(int(i) for i in S)
img = cv.imread('image.png')
print(img.shape)
for i in range(0,6):
    #text='('+str(p[i*4])+','+str(p[i*4+1])+')'
    #cv.putText(img, text, (p[i*4]-30,p[i*4+1]+15), cv.FONT_HERSHEY_TRIPLEX, 0.3, (0,0,255), 1)
    #text='('+str(p[i*4+2])+','+str(p[i*4+3])+')'
    #cv.putText(img, text, (p[i*4+2]-30,p[i*4+3]-5), cv.FONT_HERSHEY_TRIPLEX, 0.3, (0,0,255), 1)
    for j in range(0,2):
        cv.line(img ,(p[i*4+j*2],p[i*4+1]), (p[i*4+j*2],p[i*4+3]), (0,0,255), thickness = 3)
    for j in range(0,2):
        cv.line(img ,(p[i*4],p[i*4+1+j*2]), (p[i*4+2],p[i*4+1+j*2]), (0,0,255), thickness = 3)
cv.imshow('Car', img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.waitKey(1)
#print(img.shape)


