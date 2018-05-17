# Hand Sign Recognition using OpenCV
# Done by: Shreya Vishwanath Rao, Shreya Sudip, Vishnu Raghunath and Siddharth Bapat
# Version 1.0: 6/8/2018

import numpy as np
import cv2
import math
import re

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(gray,(250,250),(75,75),(0,255,0),0)
    cv2.rectangle(gray,(215,75),(250,250),(0,255,0),1)#thumb
    cv2.rectangle(gray,(180,75),(215,140),(0,255,0),1)#index
    cv2.rectangle(gray,(145,75),(180,140),(255,0,0),0) #middle
    cv2.rectangle(gray,(110,75),(145,140),(255,0,255),0) #ring
    cv2.rectangle(gray,(75,222),(110,140),(255,0,255),0) #little
    cv2.line(gray,(75,168),(110,168),(255,0,255),0) 
    cv2.line(gray,(75,195),(110,195),(255,0,255),0)
    cv2.rectangle(gray,(75,222),(145,250),(255,0,255),0)
    crop_img = gray[75:250, 75:250]
    value = (15,15)
    blurred = cv2.GaussianBlur(crop_img, value, 0)
    _, thresh1 = cv2.threshold(blurred, 05, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)

    #finding contours
    _, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnt=contours[ci]

    #finding centroid
    M=cv2.moments(cnt)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
    hull_val = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    cv2.drawContours(drawing,[hull_val],0,(0,0,255),0)
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    defects = cv2.convexityDefects(cnt,hull)
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)

    #finding orientation
    H=0
    if (hull_val[0][0][0]>=172 and hull_val[0][0][0]<=175):
        H=1
    
    #spliting coordinates
    split_hull_val=np.array_split(hull_val, len(hull_val))
    int_hull_val=[]
    for i in range(len(split_hull_val)):
        int_hull_val.append([])
        hull_str=str(split_hull_val[i])
        temp=[int(s) for s in re.findall(r'\b\d+\b', hull_str)]
        int_hull_val[i].append(temp)

    #plotting points on the convex hull
    for i in range(len(int_hull_val)):
        cv2.circle(crop_img,(int_hull_val[i][0][0],int_hull_val[i][0][1]),3,[255,255,255],-1)

    #drawing contours
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(crop_img,start,end,[0,255,0],2)

    #finger detection
    L=R=M=I=T=0
    if ((w<h) and H==0):
        for j in range(len(int_hull_val)):
            if (int_hull_val[j][0][0]>=140) and (int_hull_val[j][0][1]<=cy):
                T=1
            elif (int_hull_val[j][0][0]>=105 and int_hull_val[j][0][0]<140) and (int_hull_val[j][0][1]<=65):
                I=1
            elif (int_hull_val[j][0][0]>=70 and int_hull_val[j][0][0]<105) and (int_hull_val[j][0][1]<=65):
                M=1
            elif (int_hull_val[j][0][0]>=35 and int_hull_val[j][0][0]<70) and (int_hull_val[j][0][1]<=65):
                R=1
            elif (int_hull_val[j][0][0]>=0 and int_hull_val[j][0][0]<35) and (int_hull_val[j][0][1]>=65 and int_hull_val[j][0][1]<=cy):
                L=1
    elif ((w>h) and H==1):
        for j in range(len(int_hull_val)):
            if (int_hull_val[j][0][0]>=0 and int_hull_val[j][0][0]<=175) and (int_hull_val[j][0][1]<65):
                T=1
            elif (int_hull_val[j][0][0]>=0 and int_hull_val[j][0][0]<=35) and (int_hull_val[j][0][1]>=65 and int_hull_val[j][0][1]<93):            
                I=1
            elif (int_hull_val[j][0][0]>=0 and int_hull_val[j][0][0]<=35) and (int_hull_val[j][0][1]>=93 and int_hull_val[j][0][1]<120):
                M=1
            elif (int_hull_val[j][0][0]>=0 and int_hull_val[j][0][0]<=35) and (int_hull_val[j][0][1]>=120 and int_hull_val[j][0][1]<147):
                R=1
            elif (int_hull_val[j][0][0]>=0 and int_hull_val[j][0][0]<=70) and (int_hull_val[j][0][1]>=147 and int_hull_val[j][0][1]<=175):
                L=1

    #creating an array 'alpha'
    alpha=[]
    for i in range(2): #level 1
        alpha.append([])
        alpha.append([])
        for j in range(2): #level 2
            alpha[i].append([])
            alpha[i].append([])
            for k in range(2): #level 3
                alpha[i][j].append([])
                alpha[i][j].append([])
                for l in range(2): #level 4
                    alpha[i][j][k].append([])
                    alpha[i][j][k].append([])
                    for m in range(2):
                        alpha[i][j][k][l].append([])
                        alpha[i][j][k][l].append([])

    fo1=open("alpha.txt",'r')
    A=fo1.read()
    index=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        alpha[i][j][k][l][m].append(A[index])
                        index=index+1
    
    fo2=open("digit.txt",'r')
    digit=fo2.read() #creates and initializes array named 'digit'
    
    b="     "
    if ((w<h) and H==0): #checking orientation
        a=str(alpha[T][I][M][R][L])
        letter=a[2]
        if letter=='*':
            cv2.putText(gray,"%s" % b, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(gray,"%s" % letter, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    elif ((w>h) and H==1):
        sum=T+(2*I)+(3*M)+(4*R)+(5*L)-2
        num=str(digit[sum])
        if (num=='*'):
            cv2.putText(gray,"%s" % b, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(gray,"%s" % num, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    else:
        cv2.putText(gray,"Hello World!!!", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

    #engine.runAndWait()

    cv2.imshow('end', crop_img)
    cv2.imshow('Gesture', gray)
    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)

    # Display the resulting frame   
    k = cv2.waitKey(10)
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
