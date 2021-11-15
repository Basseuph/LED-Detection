from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# reference coordinates are taken from realTraining2
import numpy.linalg


def getRoiByImage(img, H):
    # from top left

    # 1049x654
    scale_x = img.shape[1] / 1049
    scale_y = img.shape[0] / 654

    # p1 = np.array([20,20,1])

    # img = cv2.circle(img, p1[0:1], 5, (0,0,255), 3 )

    # r_p1 = np.matmul(H, p1)
    # plt.imshow(img)
    # plt.show()

    # return r_p1

    led1_top_left = np.rint(np.array([1 * scale_x, 69 * scale_y, 1])).astype(int)
    led1_bottom_right = np.rint(np.array([12 * scale_x, 105 * scale_y, 1])).astype(int)

    led2_top_left = np.rint(np.array([1 * scale_x, 115 * scale_y, 1])).astype(int)
    led2_bottom_right = np.rint(np.array([9 * scale_x, 147 * scale_y, 1])).astype(int)

    led1 = img[led1_top_left[1]:led1_bottom_right[1], led1_top_left[0]:led1_bottom_right[0]]
    led2 = img[led2_top_left[1]:led2_bottom_right[1], led2_top_left[0]:led2_bottom_right[0]]

    # led1 = cv2.cvtColor(led1, cv2.COLOR_BGR2GRAY)

    # _, thresh = cv2.threshold(led1, 250, 255, cv2.THRESH_BINARY)

    # cv2.imshow("thresh", thresh)

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    # ax1.imshow(led1)
    # ax2.imshow(led2)
    # plt.show()

    # cv2.imshow("Led1", led1)
    # cv2.imshow("Led2", led2)
    # cv2.waitKey(0)
    return led1, led2


def get_roi_by_dest_corners(img, H):
    # 1024x768 reference.jpg, cropped 432x283
    # coordinates from reference.jpg, relative to top left corner
    #measured_corners = np.array([[0, 32, 1], [5, 43, 1], [0, 51, 1], [4, 62, 1]])
    led_center = np.array([[2, 38, 1], [2, 57, 1]])
    measured_hw = (432, 283)

    #scale_x = abs(crn_pts_src[0][0] - crn_pts_src[2][0]) / measured_hw[0]
    #scale_y = abs(crn_pts_src[0][1] - crn_pts_src[1][1]) / measured_hw[1]



    for corner in led_center:
        corner[0] = corner[0]# * scale_x
        corner[1] = corner[1]# * scale_y
        t_corner = np.matmul(H, corner)
        corner[0] = t_corner[0]
        corner[1] = t_corner[1]

    leds = led_by_circle_coordinates(img, led_center.astype(int), 5)

    #cv2.rectangle(img, (transformed_corners[0][0], transformed_corners[0][1]),
    #              (transformed_corners[1][0], transformed_corners[1][1]), (255, 0, 0), 2)
    #cv2.rectangle(img, (transformed_corners[2][0], transformed_corners[2][1]),
    #              (transformed_corners[3][0], transformed_corners[3][1]), (255, 255, 0), 2)

    cv2.imshow("Led1", leds[0])
    cv2.imshow("Led2", leds[1])
    #cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    #cv2.imshow("Result", img)
    #cv2.waitKey(0)
    return leds[0], leds[1]


def led_by_circle_coordinates(img, circle_centers, r):
    leds = []
    for center in circle_centers:
        top_left = (center[0] - r, center[1] - r)
        bottom_right = (center[0] + r, center[1] + r)
        led = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        leds.append(led)

    return leds
