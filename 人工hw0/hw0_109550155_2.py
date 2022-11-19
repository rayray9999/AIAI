#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:58:43 2022

@author: raytm9999
"""

import cv2 as cv
import numpy as np
#cv.startWindowThread()
cp=cv.VideoCapture('video.mp4')
LF = None
while (cp.isOpened()):
    isTrue,frame=cp.read()
    if(LF is None): LF=frame
    fD =cv.absdiff(frame,LF)
    LF=frame
    b=np.zeros(frame.shape[:2],dtype ="uint8")
    fD = cv.cvtColor(fD, cv.COLOR_BGR2GRAY)
    #isTrue, fD =cv.threshold(fD, 30, 255, cv.THRESH_BINARY)
    fD=cv.merge([b,fD,b])
    if isTrue: 
        com =np.hstack((frame,fD))
        cv.imshow('Video', com)
        # 讀取過程中若按下 q 則離開
    if cv.waitKey(1) & 0xFF==ord('q'):
        break 
    
cp.release()
cv.destroyAllWindows()
#cv.waitkey(1)