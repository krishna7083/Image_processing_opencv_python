from collections import deque
import numpy as np
import argparse
import imutils
import cv2
'''handle parsing our command line arguments
If this switch is supplied, then OpenCV will grab a pointer to the video file and read frames from it..
Otherwise, if this switch is not supplied, then OpenCV will try to access our webcam'''
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())
 
# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117),'orange':(0, 50, 80)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255),'orange':(20,255,255)}
 
# define standard colors for circle around the object
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0),'orange':(0,140,255)}
# here we are intilaziing our video webcam object 
if not args.get("video", False):
    cam= cv2.VideoCapture(0)         
   
#otherwise, grab a reference to the video file
else:
    cam = cv2.VideoCapture(args["video"])

while True:
    # it will read the current frame 
    (grabbed, frame) = cam.read()
    #if we are viewing a video and we did not grab a frame , then we have reached the end of the video
    if args.get("video") and not grabbed:
        break
 
   
    # now  resize the frame, blur it...and convert it to the HSV
    frame = imutils.resize(frame, width=700)
 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for key, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #opening is just another name of erosion followed by dilation. It is useful in removing noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #Closing is reverse of Opening, Dilation followed by Erosion.
                                                                #It is useful in closing small holes inside the foreground objects, or small black points on the object.
               
        # find contours in the mask and initialize the current
        
        cnt = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
       
        # only proceed if at least one contour was found
        if len(cnt) > 0:
            # find the largest contour in the mask, then use
            cont = max(cnt, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(cont)
            M = cv2.moments(cont)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
       
            #only proceed if the radius meets a minimum size. Correct this value for your obect's size
            if radius > 0.5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame,key + " ball", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
 
     
   
    cv2.imshow("Frame", frame)
   
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
 

cam.release()
cv2.destroyAllWindows()
