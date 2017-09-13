import cv2

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS,0)


#create cascade classifier
cl_rpalm=cv2.CascadeClassifier('rpalm.xml')

framecount=0
p0=p1=p2=p3=()
global flag
global lose_track
flag=0
lose_track=0
while(cap.isOpened()):
	ret,frame=cap.read()
	
	if ret:
		
		gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

		org=(int(0.04*gray.shape[0]),int(0.04*gray.shape[1]))
		rpalms=cl_rpalm.detectMultiScale(gray,1.05,3)
		
		if (len(rpalms) != 0):
									
			x,y,width,height=rpalms[0]

			if flag==0:
				p0=(int(x+0.5*width),int(y+0.5*height)) if not p0 else p0
				
			if flag==1:
				p1=(int(x+0.5*width),int(y+0.5*height)) if p0 and not p1 else p1
				
			if flag==2:
				p2=(int(x+0.5*width),int(y+0.5*height)) if p0 and p1 and not p2 else p2

			if flag==3:
				p3=(int(x+0.5*width),int(y+0.5*height)) if p0 and p1 and p2 and not p3 else p3

			
			if p0:cv2.circle(frame,p0,2,(255,0,0),3);cv2.putText(frame,'p0',p0,1,1,(0,0,255))

			if p1:cv2.circle(frame,p1,2,(255,0,0),3);cv2.putText(frame,'p1',p1,1,1,(0,0,255))

			if p2:cv2.circle(frame,p2,2,(255,0,0),3);cv2.putText(frame,'p2',p2,1,1,(0,0,255))

			if p3:cv2.circle(frame,p3,2,(255,0,0),3);cv2.putText(frame,'p3',p3,1,1,(0,0,255))


			if flag==0:
				print 'flag=0 :',p0,p1,p2,p3
				if p1:
					cv2.line(frame,p1,p2,(0,255,0),2)
					cv2.line(frame,p2,p3,(0,255,0),2)
					cv2.line(frame,p3,p0,(0,255,0),2)


			if flag==1:
				print 'flag=1 :',p0,p1,p2,p3
				if not p2:
					cv2.line(frame,p0,p1,(0,255,0),2)
				else:
					cv2.line(frame,p2,p3,(0,255,0),2);cv2.line(frame,p3,p0,(0,255,0),2);cv2.line(frame,p0,p1,(0,255,0),2)
					

			if flag==2:
				print 'flag=2 :',p0,p1,p2,p3
				if not p3:
					cv2.line(frame,p0,p1,(0,255,0),2);cv2.line(frame,p1,p2,(0,255,0),2)
				else:
					cv2.line(frame,p3,p0,(0,255,0),2);cv2.line(frame,p0,p1,(0,255,0),2);cv2.line(frame,p1,p2,(0,255,0),2)

			if flag==3:
				print 'flag=3 :',p0,p1,p2,p3
				cv2.line(frame,p0,p1,(0,255,0),2);cv2.line(frame,p1,p2,(0,255,0),2);cv2.line(frame,p2,p3,(0,255,0),2)
				
			
			
			
						
			cv2.rectangle(frame,(x,y),(x+width,y+height),(0,0,255),2)
			

			flag+=1
		
			if p0 and p1 and p2 and p3:
				if flag==4 : 
					flag=0;p0=()
				elif flag==1 :
					p1=()
				elif flag==2 : 
					p2=()
				elif flag==3 : 
					p3=()
		else:
			lose_track+=1


		if lose_track==3:lose_track=0;p0=p1=p2=p3=();flag=0
					
		framecount+=1
		cv2.putText(frame,'ftp : '+str(framecount),org,1,1,(0,0,255))
		cv2.imshow('frame',frame)
		
		key=cv2.waitKey(10)
		
		if (key & 0xff==ord('q')):
			break
		

		
		
cap.release()
cv2.destroyAllWindows()



