#-*-coding:utf-8-*-
#
# 摄像头三状态： shieldFlag -- 遮挡   noshieldFlag -- 无遮挡  bishieldFlag -- 两者之间
# 无遮挡状态下每20帧更新一次背景
# 遮挡状态下不对背景进行更新
# 介于遮挡和非遮挡之间时，保持该状态300帧判断为背景发生了变化，更新背景
#

import cv2
import sys
import numpy as np

#set the thresh
thresh=eval(sys.argv[1]) if len(sys.argv) ==2 else 50
#open the camera
cap=cv2.VideoCapture(0)

#set the frame property
cap.set(cv2.CAP_PROP_FOCUS,0)
height = int(cap.get(3))
width  = int(cap.get(4))


#if fail to open camera , quit
if not cap.isOpened():
	print 'fail to open camera'
	sys.exit(0)


#define the global variable and assign
global ori_r
global ori_g
global ori_b

global res_r
global res_g
global res_b

global flag

res_r = 0
res_g = 0
res_b = 0


# define the original value of flag
shieldFlag=False
noshieldFlag=True
bishieldFlag=False


# count the frame
contextChCount = 0
framecount = 0

while(True):
	
	ret,src=cap.read()
	
	# if rgb perform not very well, can change the color map to hsv , YUV
	r,g,b=cv2.split(src)
	

	# update the context
	if  framecount %20 ==0 and not shieldFlag and noshieldFlag and not bishieldFlag:
		
	
			ori_r = r
			ori_g = g
			ori_b = b

	elif bishieldFlag and contextChCount % 200 ==0 :

			ori_r = r
			ori_g = g
			ori_b = b
			 
	
	# compute the difference between the context frame and current frame
	res_r = np.sum(cv2.absdiff(ori_r , r))/(r.shape[0]*r.shape[1])
	res_g = np.sum(cv2.absdiff(ori_g , g))/(g.shape[0]*g.shape[1])
	res_b = np.sum(cv2.absdiff(ori_b , b))/(b.shape[0]*b.shape[1])

	#condition = ( res_r > thresh and res_g > thresh ) or ( res_r > thresh and res_b > thresh ) or ( res_b > thresh and res_g > thresh ) 
	
	
	
	condition = res_r > thresh or res_g > thresh or res_b > thresh

	framecount += 1

	contextChange = res_r < 10 or res_g < 10 or res_b < 10
	
	if condition :
		print str(framecount).center(50,'-')
		print 'detect shileld happened !!!'

		shieldFlag=True
		noshieldFlag=False
		bishieldFlag=False

		contextChCount += 1
	elif contextChange:

		noshieldFlag=True
		shieldFlag=False
		bishieldFlag=False

		contextChCount = 0
	else :
		bishieldFlag=True
		noshieldFlag=False
		shieldFlag=False

		contextChCount += 1
	


	
	cv2.putText(src,'count : ' +str(framecount),(50,50),0,1,(0,255,255),2)
	cv2.imshow('demo',src)
	key=cv2.waitKey(10)
	if key & 0xff==ord('q'):
		break

cv2.destroyAllWindows()
cap.release()


	
	
