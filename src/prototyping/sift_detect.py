from __future__ import print_function
from cv2 import cv2
import numpy as np
import os

from matplotlib import pyplot as plt

from src.prototyping.roi_dedection import get_roi_by_dest_corners


def homography_by_sift(ref_img, im_img):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(im_img, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.80 * n.distance:
            good.append(m)

    M = None
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


        matchesMask = mask.ravel().tolist()
        h, w, d = ref_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        im_img = cv2.polylines(im_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(ref_img, kp1, im_img, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()
    return M


if __name__ == '__main__':
    homography_by_sift(cv2.imread(os.path.join("referenceCropped.jpg"), cv2.IMREAD_COLOR),
                       cv2.imread(os.path.join("resources", "realTraining2.jpg"), cv2.IMREAD_COLOR))
