'''                      

Based on original code by Anirban Kar:  

modified by Santosh Premi Adhikari   

'''

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_smile.xml')

# For each person, enter one numeric face id

face_id = input('\n enter user id and press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    img = cv2.flip(img, 1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 
        
        eyes = eyeCascade.detectMultiScale(
            gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smile = smileCascade.detectMultiScale(
            gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
        
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(gray, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        


        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
         break
    elif count >= 40: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


