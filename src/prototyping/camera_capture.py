from time import sleep

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.prototyping.destinationPoints import show_leds, detect_status

corners = []
img = cv2.imread("./resources/test3.jpg")

def capture_camera():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./resources/piOnOff.mp4")



    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_click)
    print("Please click on the first corner")

    old_s1 = False
    old_s2 = False

    while True:
        ret, rec = cap.read()

        #Loop video if finished
        if not ret:
            cap.release()
            cap = cv2.VideoCapture("./resources/piOnOff.mp4")
            continue

        global img
        img = rec
        cv2.imshow("Camera", rec)

        if len(corners) == 4:
            s1, s2 = detect_status(corners, img)

            if s1 != old_s1:
                print("LED1 on: " + str(s1))
                old_s1 = s1
            if s2 != old_s2:
                print("LED2 on: " + str(s2))
                old_s2 = s2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.waitKey(1000000)
    cap.realease()
    cv2.destroyAllWindows()

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        corner = [x, y]
        corners.append(corner)
        print("Please click on the next corner if not done yet")





capture_camera()