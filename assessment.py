import cv2
import pydicom
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pyefd
import csv
from skimage.feature import greycomatrix, greycoprops
import pywt
from skimage import io
from MLP_classifier import MLP


    #ds=pydicom.dcmread('\\DICOM\\canvas\\13086')

#path = "'E:\\DICOM\\canvas\\img.png'"
img = cv2.imread('E:\\DICOM\\canvas\\roi.png')
#gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#thresh= 10
#thresh, binary = cv2.threshold(gray, thresh, maxval=255, type=cv2.THRESH_BINARY) 
#thresh, mask = cv2.threshold(gray, thresh, maxval=1, type=cv2.THRESH_BINARY)
#origin = cv2.imread('E:\\DICOM\\canvas\\blur.png', cv2.IMREAD_GRAYSCALE)
#adjusted = origin*mask
    #cv2.imshow("ADJUSTED",adjusted)
    #calculate glcm
glcm = greycomatrix(img, 
                       distances=[1, 2], 
                       angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                       symmetric=True,
                       normed=True)
glcm_br = glcm[1:, 1:, :, :] #remove black pixel
glcm_br_norm = np.true_divide(glcm_br, glcm_br.sum(axis=(0, 1))) #hình thức hóa ma trận glcm
np.set_printoptions(threshold=1000, precision=4) #quy định cách biểu diễn số dạng float, precision = 4 nghĩa là lấy 4 chữ số sau dấu phẩy
contrast = greycoprops(glcm_br_norm, 'contrast')[0][0]
energy = greycoprops(glcm_br_norm, 'energy')[0][0]
homogeneity = greycoprops(glcm_br_norm, 'homogeneity')[0][0]
correlation = greycoprops(glcm_br_norm, 'correlation')[0][0]
dissimilarity = greycoprops(glcm_br_norm, 'dissimilarity')[0][0]
ASM = greycoprops(glcm_br_norm, 'ASM')[0][0] 
pixels = cv2.countNonZero(img)
cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#lấy tất cả điểm trên countour. cv2.CHAIN_APPROX_SIMPLE: hạn chế số điểm trên countour

if len(cnts) == 2: #phan biet hinh 2d va 3d
        cnts = cnts[0] 
else:
        cnts = cnts[1] 
total = 0
for c in cnts:
       x,y,w,h = cv2.boundingRect(c)
       mask = np.zeros(img.shape, dtype=np.uint8)
       cv2.fillPoly(mask, [c], [255,255,255])
       mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
       pixels = cv2.countNonZero(mask)
       total += pixels
       dt=pixels
       #dt = pixels*ds.PixelSpacing[0]*ds.PixelSpacing[1]
       s = round(dt, 2)
       #cv2.putText(img, '{}'.format(s)+' mm2', (x,y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
for c in cnts: 
  	# compute the center of the contour
      M = cv2.moments(c)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
      l= len(c)
	# draw the contour and center of the shape on the image
      color = cv2.cvtColor(adjusted,cv2.COLOR_GRAY2RGB)
      cv2.drawContours(color, [c], -1, (255, 255, 255), 1)
      cv2.circle(color, (cX, cY), 3, (0, 0, 255), -2)
      #cv2.putText(adjusted, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      cv2.imshow('contour',color)
      #cv = len(c)*ds.PixelSpacing[0]
      cv = len(c)
      cv= round(cv, 2)
      #cv2.putText(img, '{}'.format(cv)+' mm', (x,y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    #print(len(c))# len(c)xPixelSpacing = chu vi
b = []
c0= (cv**2)/(dt)
    #phan tich fourier
N = 0
for i in range(0, len(c)): #chạy từ 0 đến len(c)
        N = N + 1
        p= cnts[0][i]
        d = math.sqrt(((p[0][0]-cX)**2)+((p[0][1]-cY)**2))
        d = round(d, 8)
        #a.append(i)
        b.append(d)
for i in range(0,len(b)):
        b[i] = str(b[i])
        with open('C:\\Users\\ADMIN\\Downloads\\csv\\fft.csv', mode='a+',newline='') as train:
         train = csv.writer(train)
         train.writerow([b[i]])   
    #fourier
fs = N
    
    
    
    
an = bn = cn = dn = 0
coeffs = []
order=100
for c in cnts:
    coeffs=pyefd.elliptic_fourier_descriptors(np.squeeze(c), order, normalize=False)
    for i in range(order):
            an = an+coeffs[i][0]
            an = abs(an)
            bn = bn+coeffs[i][1]
            bn = abs(bn)
            cn = cn+coeffs[i][2]
            cn = abs(cn)
            dn = dn+coeffs[i][3]
            dn = abs(dn)
an=an/order
bn=bn/order
cn=cn/order
dn=dn/order
with open('C:\\Users\\ADMIN\\Downloads\\csv\\test.csv', mode='a+',newline='') as train:
        train = csv.writer(train)
        train.writerow([dt,cv,c0, bn,cn, contrast, energy, homogeneity,dissimilarity, ASM])
