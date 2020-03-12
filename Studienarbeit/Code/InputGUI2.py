import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
cam = cv.VideoCapture(0)
cv.namedWindow("InputGUI")
img_counter = 0
while True:
    ret, img = cam.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow("InputGUI", img)
    if not ret:
        break
    k = cv.waitKey(1)
    if k%256 == 27: # ESC pressed, closing the window
        break
    elif k%256 == 32: # SPACE pressed, save image as input 
        cv.imwrite("InputGUI{}.png".format(img_counter), img)
        img_counter += 1

cam.release()
cv.destroyAllWindows()
