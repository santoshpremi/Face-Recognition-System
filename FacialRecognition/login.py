from tkinter import *
import os
import numpy as np
from PIL import Image
from cv2 import cv2

# Designing window for registration

def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("300x250")

    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()

    Label(register_screen, text="Please enter details below", bg="blue").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ")
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * ")
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command = register_user).pack()


# Designing window for login 

def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("300x250")
    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()

    global username_verify
    global password_verify

    username_verify = StringVar()
    password_verify = StringVar()

    global username_login_entry
    global password_login_entry

    Label(login_screen, text="Username * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command = login_verify).pack()

# Implementing event on register button

def register_user():

    username_info = username.get()
    password_info = password.get()

    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()

    username_entry.delete(0, END)
    password_entry.delete(0, END)

    Label(register_screen, text="Registration Success", fg="green", font=("calibri", 11)).pack()

# Implementing event on login button 

def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)

    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            main()

        else:
            password_not_recognised()

    else:
        user_not_found()

# Designing popup for login success


# Designing popup for login invalid password

def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()

# Designing popup for user not found
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()

# Deleting popups

def delete_login_success():
    login_success_screen.destroy()


def delete_password_not_recognised():
    password_not_recog_screen.destroy()


def delete_user_not_found_screen():
    user_not_found_screen.destroy()


def main():
    global screen
    screen = Tk()
    screen.geometry("700x500")
    screen.title("HOMESCREEN")
    Button(screen, text="Dataset", height="2", width="30",command=dataset).pack()
    Button(screen, text="Train", height="2", width="30",command=train).pack()
    Button(screen, text="Recognize", height="2", width="30",command=recognize).pack()
   
    screen.mainloop()   
def dataset():
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
   # Do a bit of cleanup
   print("\n [INFO] Exiting Program and cleanup stuff")
   cam.release()
   cv2.destroyAllWindows()
def train():
    path = 'C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_frontalface_default.xml")
    eyeCascade = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_eye.xml')
    smileCascade = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_smile.xml')


 
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
             faceSamples.append(img_numpy[y:y+h,x:x+w])
             ids.append(id)

        return faceSamples,ids

    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/trainer/trainer.yml') 

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
def recognize():
 recognizer = cv2.face.LBPHFaceRecognizer_create()
 recognizer.read('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/trainer/trainer.yml')
 cascadePath = "C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_frontalface_default.xml"
 eyeCascade = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_eye.xml')
 smileCascade = cv2.CascadeClassifier('C:/Users/Lenovo/PycharmProjects/OpenCV-Face-Recognition-master/FaceDetection/Cascades/haarcascade_smile.xml')

 faceCascade = cv2.CascadeClassifier(cascadePath)

 font = cv2.FONT_HERSHEY_SIMPLEX

 #iniciate id counter
 id = 0

 # names related to ids: example ==> premi: id=1,  etc
 names = ['None', 'premi', 'sushant', 'siddhartha', 'premi', ''] 

 # Initialize and start realtime video capture
 cam = cv2.VideoCapture(0)
 cam.set(3, 640) # set video widht
 cam.set(4, 480) # set video height

 # Define min window size to be recognized as a face
 minW = 0.1*cam.get(3)
 minH = 0.1*cam.get(4)

 while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vert

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

         
        
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

 #Do a bit of cleanup
 print("\n [INFO] Exiting Program and cleanup stuff")
 cam.release()
 cv2.destroyAllWindows()    
# Designing Main(first) window

def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("300x250")
    main_screen.title("Account Login")
    Label(text="Select Your Choice", bg="blue", width="300", height="2", font=("Calibri", 13)).pack()
   
    Button(text="Login", height="2", width="30", command = login).pack()
   
    Button(text="Register", height="2", width="30", command=register).pack()

    main_screen.mainloop()

main_account_screen()
