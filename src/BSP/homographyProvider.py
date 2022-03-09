from cv2 import cv2
import numpy as np
import enum
import torch
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

from BSP.BoardOrientation import BoardOrientation
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_scale(crn_pts_src, crn_pts_dst):
    """
    calculates the x scale and y scaling of the image by a set of corner points
    :param crn_pts_src: is a set of corner points of the source image,
     order of them should be LT, RT, LB, RB (L/R= Left/Right, T/B=Top/Botton)
    :param crn_pts_dst: is a set of corner points of the source image,
     order of them should be LT, RT, LB, RB (L/R= Left/Right, T/B=Top/Botton)
    :return: a tuple (scale_x, scale_y)
    """
    dist_src_x = np.linalg.norm(crn_pts_src[0], crn_pts_src[1])
    dist_src_y = np.linalg.norm(crn_pts_src[0], crn_pts_src[3])
    dist_dst_x = np.linalg.norm(crn_pts_dst[0], crn_pts_dst[1])
    dist_dst_y = np.linalg.norm(crn_pts_dst[0], crn_pts_dst[3])

    scale_x = dist_dst_x / dist_src_x
    scale_y = dist_dst_y / dist_src_y

    return (scale_x, scale_y)


def scale_point(point, scaling):
    scaled_point = (point[0] * scaling[0], point[1] * scaling[1])
    return scaled_point


def knn_match(des1, des2, distance_factor=0.65):
    """
    Calculates the matches between two descriptors using knnMatch
    :param des1:
    :param des2:
    :param distance_factor:
    :return:
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < distance_factor * n.distance:
            good.append(m)
    return good


# 1st geometricall inconsistent nearest neighbor ratio matching strategy - FGINN
# D. Mishkin, J. Matas, M. Perdoch. MODS: Fast and Robust Method for Two-View Matching. CVIU 2015
def match_fginn(desc1, desc2, kps1, kps2):
    xy1 = np.concatenate([np.array(p.pt).reshape(1, 2) for p in kps1], axis=0)
    xy2 = np.concatenate([np.array(p.pt).reshape(1, 2) for p in kps2], axis=0)

    dm = torch.cdist(torch.from_numpy(desc1.astype(np.float32)).to(device),
                     torch.from_numpy(desc2.astype(np.float32)).to(device))
    vals, idxs_in_2 = torch.min(dm, dim=1)
    # xy2, xy2 is not a typo below, because we need to have a distance between
    # keypoint in the same image
    km = torch.cdist(torch.from_numpy(xy2.astype(np.float32)).to(device),
                         torch.from_numpy(xy2.astype(np.float32)).to(device))
    mask1 = km <= 10.0
    mask2 = mask1[idxs_in_2, :]
    dm[mask2] = 100000  # some big number to mask out
    vals_2nd, idxs_in_2_2nd = torch.min(dm, dim=1)
    ratio = vals / vals_2nd
    mask = ratio <= 0.8
    idxs_in_1 = torch.arange(0, idxs_in_2.size(0))[mask]
    idxs_in_2 = idxs_in_2[mask]
    matches_idxs = torch.cat([idxs_in_1.view(-1, 1), idxs_in_2.cpu().view(-1, 1)], dim=1)
    return matches_idxs.cpu().data.numpy()


def homography_by_sift(ref_img, target_img, distance_factor=0.65, display_result=False) -> BoardOrientation:
    """
    Calculates the board orientation based on SIFT with knnMatch

    :param ref_img: The reference image for the calculation
    :param target_img: The target image for the calculation
    :param distance_factor: Influences the max distance of the matches as per Loew's ration test. A higher value means
        more distant matches are also included. The optimal value may differ based on the board and image
    :param display_result: If true the result is plotted
    :return: A BoardOrientation object which contains the homography matrix and the corners
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(target_img, None)

    homography_matrix = None
    dst = None
    # Match descriptors.
    matches = match_fginn(des1, des2, kp1, kp2)
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matches_mask = mask.ravel().tolist()
        h, w, d = ref_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        dst = cv2.perspectiveTransform(np.array([pts]), homography_matrix)[0]
        dst[1], dst[3] = dst[3], dst[1]
    else:
        # raise error
        raise ("Not enough matches are found - {}/{}".format(len(matches), 10))

    if display_result:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(ref_img, kp1, target_img, kp2, matches, None, **draw_params)
        cv2.imshow('Matching', img3)

    return BoardOrientation(homography_matrix, dst, ref_img.shape[:2])


class DescriptorType(enum.Enum):
    SIFT = 1,
    SURF = 2,
    ORB = 3,
    FAST = 4,


class MatchingType(enum.Enum):
    FLANN = 1,
    KNN = 2,
    FGINN = 3,


def decolorize(img):
    """
    Decolorizes the image
    :param img:
    :return:
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ransac_and_draw_matches_cv2(kps1, kps2, tentatives, img1, img2, display_result=False):
    """
    filters the matches using RANSAC and draws the matches
    :param display_result: TRUE if the result should be displayed
    :param kps1: are the key points of the first image
    :param kps2: are the key points of the second image
    :param tentatives: are the tentatives matches
    :param img1: is the first image
    :param img2: is the second image
    :return:
    """
    good = []
    for i in range(len(tentatives)):
        good.append(cv2.DMatch(tentatives[i, 0], tentatives[i, 1], 1))
    src_pts = np.float32([kps1[m[0]].pt for m in tentatives]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m[1]].pt for m in tentatives]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
    print(deepcopy(mask).astype(np.float32).sum(), 'inliers found')
    if H is None:
        raise Exception("No homography found")

    if display_result:
        matches_mask = mask.ravel().tolist()
        h, w, _ = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(decolorize(img1), kps1, decolorize(img2), kps2, good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()
    return H


class HomographyProvider():
    """
    A class which provides homography matrices for a given board orientation
    """

    def __init__(self, descriptor_type=DescriptorType.SIFT, matching=MatchingType.KNN, display_result=False):

        self.matcher = None
        self.detector = None
        self.display_result = display_result
        # init the detector and matcher
        self._init_detector(descriptor_type)
        self._init_matching(matching)
        self._matching_type = matching

    def _init_detector(self, descriptor_type):
        """
        Initializes the detector based on the descriptor type
        :param descriptor_type:
        :return:
        """
        if descriptor_type == DescriptorType.SIFT:
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif descriptor_type == DescriptorType.SURF:
            self.detector = cv2.xfeatures2d.SURF_create()
        elif descriptor_type == DescriptorType.ORB:
            self.detector = cv2.ORB_create()
        elif descriptor_type == DescriptorType.FAST:
            self.detector = cv2.FastFeatureDetector_create()
        else:
            raise Exception("Unknown descriptor type")

    def _init_matching(self, matching_type):
        """
        Initializes the matching based on the matching type
        :param matching_type:
        :return:
        """
        if matching_type == MatchingType.FLANN:
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        elif matching_type == MatchingType.KNN:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matching_type == MatchingType.FGINN:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise Exception("Unknown matching type")

    def _get_keypoints(self, img_1, img_2, decolorized=False):
        """
        Returns the key points and descriptors for the given images
        :param img:
        :return:
        """
        if self.detector is None:
            raise Exception("Detector not initialized")
        if decolorized:
            img_1 = decolorize(img_1)
            img_2 = decolorize(img_2)

        kp1, des1 = self.detector.detectAndCompute(img_1, None)
        kp2, des2 = self.detector.detectAndCompute(img_2, None)
        return kp1, kp2, des1, des2

    def get_homography(self, img_1, img_2, board_orientation, decolorized=False):
        """

        :param img_1:
        :param img_2:
        :param board_orientation:
        :param decolorized:
        :return:
        """
        if self.matcher is None:
            raise Exception("Matcher not initialized")
        kp1, kp2, des1, des2 = self._get_keypoints(img_1, img_2, decolorized)
        if len(kp1) < 4 or len(kp2) < 4:
            raise Exception("Not enough keypoints found")
        matches = self.matcher.knnMatch(des1, des2, k=2)