#coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2,random
import os
import numpy as np
import shutil
import platform

import argparse
import cv2
def get_ims(imgpath):
    imgpathlst=[]
    for dirpath, dirnames, filenames in os.walk(imgpath):
        # subdir=lstripstr(dirpath,imgpath)
        for filename in filenames:
            if os.path.splitext(filename)[1] in ['.jpg','.jpeg','.png']:
                imgpathlst.append(os.path.join(imgpath, dirpath, filename))
    return imgpathlst

if __name__ == '__main__':

    srcroot=r'/disks/disk1/Workspace/Project/Pytorch/FaceEdit/stylegan2-ada-pytorch-main/myout/img'


    ims=get_ims(srcroot)



    numim=len(ims)
    for i,im in enumerate(ims):
        img=cv2.imread(im)

        cv2.imshow('img',img)

        grimg=np.array(img)
        grimg[:,:,0]=0

        mattingimg = np.array(img)
        mattingimg[:,:,1]=img[:,:,0]
        mattingimg[:, :, 2] = img[:, :, 0]

        cv2.imshow('grimg',grimg)
        cv2.imshow('mattingimg', mattingimg)



        key=cv2.waitKey(0)
        if key==27:
            exit(0)

