# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import copy
import sys
import matplotlib.pyplot as plt
import math
import os
import imutils
from imutils import contours

K = 4
coefficient = 400  # tune as desired

script_dir = os.path.dirname(os.path.abspath(__file__))
picturesFolderPath = script_dir +'/TestPics'
imfiles = ["TestPics/" + f for f in os.listdir(picturesFolderPath)]

# load previously-saved weights
weights = np.load('weights_g.npy', allow_pickle=True)
parameters = np.load('parameters_g.npy', allow_pickle=True)

for idx, im in enumerate(imfiles):
    src0 = cv2.imread(im)
    nx, ny, ch = src0.shape
    src = np.reshape(src0, (nx*ny,ch))

    def gaussian(d, mean, cov):
        det_cov = np.linalg.det(cov)
        cov_inv = np.zeros_like(cov)
        diff = np.matrix(d - mean)
        for i in range(d.shape[1]):
            cov_inv[i, i] = 1/cov[i, i] 
        N = (2.0 * np.pi) ** (-len(d[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv, diff), axis=1))
        return N

    prob = np.zeros((nx*ny,K))
    likelihood = np.zeros((nx*ny,K))

    for cluster in range(K):
        prob[:,cluster:cluster+1] = weights[cluster]*gaussian(src, parameters[cluster]['mean'], parameters[cluster]['cov'])       
        likelihood = prob.sum(1)
        
    probabilities = np.reshape(likelihood,(nx,ny))
    probabilities[probabilities>np.max(probabilities)/coefficient] = 255    

    output = np.zeros_like(src0)
    output[:,:,0] = probabilities
    output[:,:,1] = probabilities
    output[:,:,2] = probabilities
    
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(output,(5,5),5)
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    epsilon = 0.1 * cv2.arcLength(contours[0], True)
    for c in contours:

        approx = cv2.approxPolyDP(c, epsilon, True)
        if(approx.shape[0] > 15):
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
            M = cv2.moments(approx)
            cX = int((M["m10"] / float(M["m00"] + 1e-5)))
            cY = int((M["m01"] / float(M["m00"] + 1e-5)))
            cv2.circle(output, (cX, cY), 7, (0, 0, 255), -1)
    cv2.imwrite(im[:-4] + "-contour.png", output)
    
# blur = cv2.GaussianBlur(output,(5,5),5)
# kernel = cv2.getGaussianKernel(ksize=15,sigma=2)
# output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)

cv2.imshow("out",output)
cv2.waitKey(0)
