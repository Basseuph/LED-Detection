import cv2
import numpy as np


def main():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.filterByConvexity = False
    params.filterByArea = False
    params.filterByInertia = True
    params.filterByCircularity = True
    params.minInertiaRatio = 0.1
    params.minCircularity = 0.1

    # Blob detector to detect led light
    detector = cv2.SimpleBlobDetector_create(params)

    vid = cv2.VideoCapture("resources/piOnOff.mp4")
    frame_exists, frame = vid.read()

    while frame_exists:
        # remove noise
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # led roi's
        leds = [frame[620:660, 980:1020, :], frame[660:700, 980:1020, :]]

        for i in range(len(leds)):
            led = leds[i]
            gray_led = cv2.cvtColor(led, cv2.COLOR_BGR2GRAY)
            # remove false positives
            gray_led = cv2.erode(gray_led, None, iterations=2)
            # fill gaps
            gray_led = cv2.dilate(gray_led, None, iterations=2)
            _, thresh_img = cv2.threshold(gray_led, 230, 255, cv2.THRESH_BINARY)
            keypoints = detector.detect(thresh_img)
            img_with_blobs = cv2.drawKeypoints(thresh_img, keypoints, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("led" + str(i), img_with_blobs)
            cv2.imshow("led" + str(i) + "_blob", thresh_img)

            if len(keypoints) > 0:
                state = "on"
            else:
                state = "off"
            print("led", i, "state", state)

        # press esc to stop
        if cv2.waitKey(1) == 27:
            break
        frame_exists, frame = vid.read()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
