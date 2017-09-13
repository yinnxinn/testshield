import numpy as np
import cv2
def gammatable(gamma):
	tabgamma=[]
	for i in xrange(256):
		temp1=(float)((i+0.5)/256)
		temp2=(float)(pow(temp1,gamma))
		tabgamma.append((int)(temp2*256-0.5))
	return tabgamma

def gamma_img(img):
	gammatab=gammatable(0.4545)
	
	#get the gray pic
	if (len(img.shape)==3):
		img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		
	
	src=np.empty(img.shape)
	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			src[i][j]=gammatab[img[i][j]]
	return src
				

