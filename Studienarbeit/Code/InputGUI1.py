import cv2 as cv

cam = cv.VideoCapture(0)
cv.namedWindow("InputGUI1")
img_counter = 1
while True:
    ret, img = cam.read()
    cv.imshow("InputGUI1", img)
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