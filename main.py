# -*-coding:utf8-*-#
import cv2
import init_cnn_5layer
import dlib
import threading
import sys
import time

if __name__ == '__main__':
	if time.gmtime().tm_mon<9 :
		name={"[0]":'wang','[1]':'xu','[2]':'liu','[3]':'zhang','unknown':'unknown'}
		face_detector=dlib.get_frontal_face_detector()

		label='unknown'
		top=-10
		left=-10
		right=0
		bottom=0
		#open the camera
		cap=cv2.VideoCapture(0)
		if (not cap.isOpened()):
			print 'fail to open camera!!'
			sys.exit(0)
	
		#first run or not
		firstrun=True
		if (firstrun):
			firstrun=False
			framecount=0
	
		while (True):
			ret,frame=cap.read()

		
		
			if (not len(frame)==0):
				framecount+=1


			gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
			try:
				faces=face_detector(gray,1)
				top=faces[0].top()
				left=faces[0].left()
				right=faces[0].right()
				bottom=faces[0].bottom()
				img=gray[top:bottom,left:right]
			
				if (framecount%3==0):
				
					label=str(init_cnn_5layer.use_CNN(img))

				cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0))
				org=(int(0.5*(right+left)),int(top))
				cv2.putText(frame,name[label],org,1,1,(0,255,0))
				
				
						
			except:
			
				print 'no human'
		
			org0=(20,20)
			cv2.putText(frame,str(framecount/10),org0,1,1,(0,255,255))
			cv2.imshow('show',frame)
			cv2.waitKey(10)
	else:
		raise 'unexpected time error , please make sure your timezone!!!'
			
		
		
		
		




