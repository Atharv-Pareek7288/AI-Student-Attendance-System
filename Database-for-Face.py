#importing required libraries
import cv2
import os #this library helps us to create a folder and place all the data of face inside it
import time
import imutils

#Directory Setup 
dataset = "Dataset"
name = str(input("Enter your Name: "))

#this command constructs a path by combinig the dataset folder and the user's name
path = os.path.join(dataset,name)

#This commands checks if there is a directory or not specified in path
if not os.path.isdir(path):
    #If there is no directory that is mentioned in path variable then it create one such direcotry
    os.mkdir(path)

#This sets the desired width and height for the captured image of face
(width, height) = (130,100)

#specifies the .xml file that contains the Haar cascade data for the frontol face
alg = "haarcascade_frontalface_default.xml"

#Load the Haar cascade file
haar_cascade = cv2.CascadeClassifier (alg)

#Starts the camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#initializes the image count
count = 1
time.sleep(2.0)
#only 30 images are to be clicked
while count<51 :
    print(count)
    _,img = cam.read()
    frame = imutils.resize(img, width=400)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #this converts the captured image color to gray
    face = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30))
    
    #it creates a rectangle around the face
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)  
        p = os.path.sep.join([path, "{}.png".format(
            str(count).zfill(5)
        )])
        cv2.imwrite (p, frame) #saves the captured images and names the captured images 
        count+=1 #increments the image counter 
    
    #shows the current frame with the detected face rectangle
    cv2.imshow("faceDetection", img)
    key = cv2.waitKey(10)
    if key == 27: 
        break
print("Image Captured Successfully")
cam.release()
cv2.destroyAllWindows