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
    mattingroot=r'/home/tao/disk1/Dataset/CelebAHairMask-HQ/V1.0/mask/mask'
    dstroot='/home/tao/disk1/Dataset/CelebAMask/CelebAMask-HQ/CelebA-HQ-img-256-matting'

    ims=get_ims(srcroot)



    numim=len(ims)
    for i,im in enumerate(ims):
        imname = os.path.basename(im)

        imkey=imname.split('.')[0]
        imkeynum=int(imkey)
        mattingim=os.path.join(mattingroot,str(imkeynum).zfill(5)+'.png')

        print(mattingim)

        img=cv2.imread(im)
        mattingimg=cv2.imread(mattingim)

        mattingimg=cv2.resize(mattingimg,(0,0),fx=0.25,fy=0.25)
        # cv2.imwrite(os.path.join(dstroot,imname),newimg)



        fusion_img=np.array(img)
        fusion_img[:,:,0]=mattingimg[:,:,0]
        # fusion_img[:, :, 0]=0

        cv2.imwrite(os.path.join(dstroot,imname.replace('.jpg','.png')),fusion_img)

        # cv2.imshow('mattingimg',mattingimg)
        # cv2.imshow('img', img)
        # cv2.imshow('fusion_img',fusion_img)

        print(str(i)+'/'+str(numim))

        # key=cv2.waitKey(0)
        # if key==27:
        #     exit(0)