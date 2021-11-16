from time import sleep, time

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.prototyping.destinationPoints import detect_status

def capture_camera():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./resources/piOnOff3.mp4")



    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    old_s1 = False
    old_s2 = False

    while True:
        ret, rec = cap.read()

        #Loop video if finished
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        global img
        img = rec
        cv2.imshow("Camera", rec)

        now = time()
        frameLimit = 30  # frame limiting

        s1, s2 = detect_status(img)

        if s1 != old_s1:
            print("LED1 on: " + str(s1))
            old_s1 = s1
        if s2 != old_s2:
            print("LED2 on: " + str(s2))
            old_s2 = s2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # sleep to maintain frame rate
        timeDiff = time() - now
        if timeDiff < 1.0 / (frameLimit):
            sleep(round(1.0 / frameLimit - timeDiff))

        cv2.waitKey(30)

    cv2.waitKey(1000000)
    cap.realease()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_camera()
