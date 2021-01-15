#IMPORTING LIBRARIES AND MODULES

import cv2
import numpy as np
from time import sleep

#VARIABLES TO BE USED
large_min=80 
altitude_min=80 

offset=6 

pos_lin=550  

delay= 60 

detec = []
vehicle_count= 0

#FUNCTION

def job_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

'''To capture a video, you need to create a VideoCapture object.
Its argument can be either the device index or the name of a video file(as in this case).
Device index is just the number to specify which camera.
Add the video name. After that, we can capture frame-by-frame.
At the end, capture is released.'''

cap = cv2.VideoCapture('video.mp4')

'''Background subtraction is a major preprocessing steps in many vision based applications.
Technically, we need to extract the moving foreground from static background.
OpenCV provides us 3 types of Background Subtraction algorithms:-

1.BackgroundSubtractorMOG
2.BackgroundSubtractorMOG2
3.BackgroundSubtractorGMG'''

#initialising subtractor
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    #capture frame by frame

    temp = float(1/delay)

    sleep(temp)
    #Operations on the frame

    # Using cv2.cvtColor() method 
    # Using cv2.COLOR_BGR2GRAY color space conversion code
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    '''GaussianBlur is an implementation of a blur effect using a Gaussian
    convolution kernel, with a configurable radius.'''
    
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtractor.apply(blur)

    '''Morphological operations are a set of operations that process images based on shapes.
    They apply a structuring element to an input image and generate an output image.
    The most basic morphological operations are two: Erosion and Dilation'''

    dilat = cv2.dilate(img_sub,np.ones((5,5)))

    # defining the kernel i.e. Structuring element 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # defining the opening function  
    # over the image and structuring element
    dilated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)

    '''OpenCV has findContour() function that helps in extracting the contours from the image.
    It works best on binary images, so we should first apply thresholding
    techniques, Sobel edges, etc.'''
    contour,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos_lin), (1200, pos_lin), (255,127,0), 3)

    # Searching through every region selected to  
    # find the required polygon(rectangle).

    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contour = (w >= large_min) and (h >= altitude_min)
        if not validar_contour:
            continue

        # Using the rectangle() function to create a rectangle.
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

        #displaying 'CAR' as heading above the frame
        cv2.putText(frame1, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        center = job_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        
        #Counting the vehicles that are being detected as they cross the line 
        for (x,y) in detec:
            if y<(pos_lin+offset) and y>(pos_lin-offset):
                
                #incrementing vehicle counter
                vehicle_count+=1
                
                cv2.line(frame1, (25, pos_lin), (1200, pos_lin), (0,127,255), 3)  
                detec.remove((x,y))

                #Displaying the number of vehicles detected in the output window
                print("vehicle is detected : "+str(vehicle_count))        

    #Adding the text using putText() function    
    cv2.putText(frame1, "VEHICLE COUNT : "+str(vehicle_count), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)

    # Displaying the image  
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilated)

    if cv2.waitKey(1) == 27:
        break
    
# De-allocate any associated memory usage 
cv2.destroyAllWindows()

# Close the window / Release webcam
cap.release()
