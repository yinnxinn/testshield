# -*-coding:utf8-*-#
import cv2
import init_cnn_5layer
import dlib
import threading
import sys

if __name__ == '__main__':
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
			if (not len(faces)==0):
				imgs=[]
				orgs=[]
				for i in xrange(len(faces)):
			
					top=faces[i].top()
					left=faces[i].left()
					right=faces[i].right()
					bottom=faces[i].bottom()
					cv2.rectangle(frame,(left,top),(right,bottom),(255,0,0))
					
					imgs.append(gray[top:bottom,left:right])
					orgs.append((int(0.5*(right+left)),int(top)))
				
				if (framecount%3==0):
					
					label=[str(init_cnn_5layer.use_CNN(img)) for img in imgs]

				
				for x in xrange(len(orgs)):	
					cv2.putText(frame,name[label[x]],orgs[x],1,1,(0,255,0))
										
		except:
			
			print 'no human'
		
		org0=(20,20)
		print '.'*30
		print label
		
		cv2.putText(frame,str(framecount/10),org0,1,1,(0,255,255))
		cv2.imshow('show',frame)
		cv2.waitKey(10)	
		
		
		
		
		




