import cv2

cv2.namedWindow('camera')

cam = cv2.VideoCapture(1)

while True:
    ret, img = cam.read()
    if ret:
        cv2.imshow('camera', img)
        cv2.waitKey(1)
