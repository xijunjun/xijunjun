1#coding:utf-8

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import os
import numpy as np
import shutil
import platform

import argparse
import cv2
import torch

from facexlib.alignment import init_alignment_model, landmark_98_to_68
from facexlib.detection import init_detection_model
from facexlib.visualization import visualize_alignment





if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    align_net = init_alignment_model('awing_fan')
    det_net = init_detection_model('retinaface_resnet50', half=False)

    face_size=144
    face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                   [201.26117, 371.41043], [313.08905, 371.15118]])
    face_template = face_template * (face_size / 512.0)

    while 1:
        with torch.no_grad():
            ret, frame = cap.read()
            frame=cv2.flip(frame, 1)

            bboxes = det_net.detect_faces(frame, 0.97)
            for box in bboxes:
                # print (box)
                rct=box[0:4].astype(np.int)
                land=box[5:5+10].reshape((5,2)).astype(np.int)

                cv2.rectangle(frame, (rct[0], rct[1]), (rct[2], rct[3]), (0, 0, 255), 2)
                for pt in land:
                    print (pt)
                    cv2.circle(frame,(pt[0],pt[1]),3,(255,0,0),-1,-1)

                affine_matrix = cv2.estimateAffinePartial2D(land, face_template, method=cv2.LMEDS)[0]
                cropped_face = cv2.warpAffine(frame, affine_matrix, (144,144), borderMode=0,borderValue=(135, 133, 132))  # gray
                cv2.imshow("cropped_face", cropped_face)

            landmarks = align_net.get_landmarks(frame)
            landmarks=landmarks.astype(np.int)



            # print (landmarks)
            #
            # for pt in landmarks:
            #     cv2.circle(frame,(pt[0],pt[1]),3,(255,0,0),-1,-1)


            cv2.imshow("capture", frame)
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()




    # align_net = init_alignment_model('awing_fan')
    #
    # imtxt='/disks/disk0/Dataset/FFHQ/image1024x1024/images1024x1024/ffhq_1024_imlist.txt'
    # dsttxt='/disks/disk0/Dataset/FFHQ/image1024x1024/images1024x1024/ffhq_1024_landmark.txt'
    #
    # with open(imtxt,'r') as f:
    #     lines=f.readlines()
    #
    # allano=''
    #
    #
    #
    #
    #
    # for i,line in enumerate(lines):
    #     impath=line.rstrip('\n')
    #     img=cv2.imread(impath)
    #     landmarks = align_net.get_landmarks(img)
    #     landmarks=landmarks.astype(np.int)
    #
    #     # print (landmarks)
    #
    #     oneano=impath+' 98'
    #     for pt in landmarks:
    #         # print (pt)
    #         # cv2.circle(img,(pt[0],pt[1]),3,(255,0,0),-1,-1)
    #         oneano+=' '+str(pt[0])+' '+str(pt[1])
    #     allano+=oneano+'\n'
    #
    #
    #     # cv2.imshow('img',img)
    #     # key=cv2.waitKey(0)
    #     # if key==27:
    #     #     exit(0)
    #     # if i==10:
    #     #     break
    #     print (i)
    # allano=allano.rstrip('\n')
    # with open(dsttxt,'w') as f:
    #     f.writelines(allano)
