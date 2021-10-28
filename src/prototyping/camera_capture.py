from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.prototyping.destinationPoints import show_leds

index = 1
done = False
messages = ["Top left", "Top right", "Bottom left", "Bottom right"]
corners = []


def capture_camera():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_click)

    print("Please click on the " + messages[0] + " corner")

    while True:
        cv2.imshow("Camera", img)

        if done:
            show_leds(corners, img)
            break


def on_click(event, x, y, flags, param):
    corner = [x, y]
    corners.append(corner)

    if index < 3:
        print("Please click on the " + messages[index] + " corner")
        index += 1
    else:
        print("Done!")
        done = True