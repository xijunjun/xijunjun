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

    srcroot=r'/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256'
    dstroot=r'/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-128'

    ims=get_ims(srcroot)

    numim=len(ims)
    for i,im in enumerate(ims):
        imname = os.path.basename(im)
        img=cv2.imread(im)
        newimg=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
        cv2.imwrite(os.path.join(dstroot,imname),newimg)
        print(str(i)+'/'+str(numim))
        if i==2000:
            exit(0)