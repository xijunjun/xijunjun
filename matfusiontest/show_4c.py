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

    srcroot=r'/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-4c'


    ims=get_ims(srcroot)



    numim=len(ims)
    for i,im in enumerate(ims):
        imgori=cv2.imread(im,cv2.IMREAD_UNCHANGED)
        h,w,c=imgori.shape
        img=np.zeros((h,w,3),imgori.dtype)
        img[:,:,:]=imgori[:,:,0:3]


        cv2.imshow('img',img)



        mattingimg = np.array(img)
        mattingimg[:, :, 0] = imgori[:, :, -1]
        mattingimg[:,:,1]=imgori[:,:,-1]
        mattingimg[:, :, 2] = imgori[:, :, -1]

        cv2.imshow('mattingimg', mattingimg)



        key=cv2.waitKey(0)
        if key==27:
            exit(0)

