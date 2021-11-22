import os

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# source https://blog.ekbana.com/skew-correction-using-corner-detectors-and-homography-fda345e42e65
from src.prototyping import sift_detect
from src.prototyping.matrixcache import MatrixCache
from src.prototyping.roi_detection import getRoiByImage, get_roi_by_dest_corners

inv = None
corners = None

matrix = None


def detect_status(img):
    global matrix

    if matrix is None or matrix.check_if_outdated():
        print("Calculating new matrix")
        inv, corners = sift_detect.homography_by_sift(cv2.imread(os.path.join("referenceCropped.jpg"), cv2.IMREAD_COLOR), img)
        matrix = MatrixCache(inv, corners)

    led1, led2 = get_roi_by_dest_corners(img, matrix.matrix, matrix.corners)

    led1 = cv2.cvtColor(led1, cv2.COLOR_RGB2HSV)
    led2 = cv2.cvtColor(led2, cv2.COLOR_RGB2HSV)

    mean1 = np.nanmean(led1[..., 2])
    mean2 = np.nanmean(led2[..., 2])

    return mean1 > 165, mean2 > 165


#if __name__ == '__main__':
    #corners = [(83, 32), (1093, 104), (43, 689), (1028, 757)]  # realTraining 2
    #corners = [(223, 227), (654, 257), (205, 510), (636, 540)] #reference.jpg
    #corners = [(228, 346), (482, 163), (479, 487), (688, 247)] #angleTest.jpg
    #corners = [(111,307), (437, 300), (107, 525), (440, 517)] # realTest1
    #corners = np.array([[871, 884], [2573, 1012], [795, 1988], [2476, 2109]])  # realTraining3 rotated
    #corners = np.array([[871, 887], [2555, 1004], [801, 1992], [2484, 2109]])  # realTraining3 rotated, idealised
    #img = cv2.imread('resources/angleTest.jpg')
    #img = cv2.rotate(img, cv2.ROTATE_180)
    #show_leds(corners, img)
    #detect_status(corners, img)