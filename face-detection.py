#here we import all our necessary libaries
import cv2

#here I am defining something known as the cascade classifier, where I am inputing a haarcascade dataset of faces
face_detection = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#this is to capture my live video, and the 0 is telling the computer that I want to use my computer's cam
cap = cv2.VideoCapture(0)

#sets width and height for the video capture screen
cap.set(3,640)
cap.set(4,480)

#we use a while loop to diplay the vid, because the at the end of the day a vid is just a bunch of images repeating
while True:
    #here the computer is understanding the live video, and defining the video feed as image. Also the success variable is a boolean value here letting us know if the video was read
    success, img = cap.read()

    #here I am converting each of the frame of the bideo to grayscale, because it is easy for the computer to read
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #this is a line to detect multiscale images, meaning images of different sizes(different sizes of face).
    #the gray image here is telling it to detect multiscale iamges in the live feed
    #the 1.3 variable here is scale factor, which is to extend and decrease the size of a image so the computer can detect any scale of image
    #6 is the k neighbour, whcih is important in identifying the correct face, and reducing false detection
    faces = face_detection.detectMultiScale(img_gray, 1.3, 6)

    #for loop to use coordinates in faces and draw rectnagle around it with putting text
    for (x,y,w,h) in faces:
        rect = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
        cv2.putText(rect, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
        
    
    #dispaying the video
    cv2.imshow("Video", img)

    #if statement to break loop if q clicked
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    