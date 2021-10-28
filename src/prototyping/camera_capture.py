from time import sleep

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.prototyping.destinationPoints import show_leds

corners = []
img = cv2.imread("./resources/test3.jpg")

def capture_camera():
    #cap = cv2.VideoCapture(0)


    cv2.imshow("Camera", img)

    #cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_click)

    print("Please click on the first corner")


    cv2.waitKey(1000000)
    #cap.realease()
    cv2.destroyAllWindows()

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        corner = [x, y]
        corners.append(corner)
        print("Please click on the next corner if not done yet")

        if len(corners) == 4:
            corners[2], corners[3] = corners[3], corners[2]
            cv2.polylines(img, [np.array(corners)], False, (255, 0, 0), 4)
            cv2.imshow("Camera", img)
            corners[2], corners[3] = corners[3], corners[2]
            show_leds(corners, img)




capture_camera()