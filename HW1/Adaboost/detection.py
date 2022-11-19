import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
import utils
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    cordi=[]
    with open(dataPath+"/detectData.txt") as f:
        nump=int(f.readline())
        for k in range(nump):
            tmp=f.readline()
            tmp=tmp.split(" ")
            kk=tuple(map(int,tmp))
            cordi.append(kk)
            
    cp=cv2.VideoCapture(os.path.join(dataPath,"video.gif"))
    frame=0
    f_f=True
    while True:
        _,img=cp.read()
        frame+=1
        d_label=[]
        if img is None:
            break
        for cord in cordi:
            pt=crop(*cord,img)
            pt=cv2.resize(pt,(36,16))
            pt=cv2.cvtColor(pt,cv2.COLOR_RGB2GRAY)
            d_label.append(clf.classify(pt))
            
        for i,label in enumerate(d_label):
            if label==1:
                pos=[[cordi[i][idx],cordi[i][idx+1]] for idx in range(0,8,2)]
                pos[2],pos[3]=pos[3],pos[2]
                pos=np.array(pos,np.int32)
                cv2.polylines(img,[pos],color=(0, 255, 0),isClosed=True)
        if f_f:
            f_f=False
            cv2.imwrite(f"Adaboost_first_frame.png", img)
        with open(f"Adaboost_pred.txt", "a") as txt:
            tp=""
            for i,label in enumerate(d_label):
                if label:
                    tp+="1"
                else:
                    tp+="0"
                if i!=len(d_label)-1:
                    tp+=" "
                else:
                    tp+="\n"
            txt.write(tp)
            
    
    # End your code (Part 4)
