import logging

import cv2 as cv
import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
import enum
import torch
from scipy.spatial import distance_matrix

from BSP.BoardOrientation import BoardOrientation


def distance_matrix_vector(anchor, positive, binary_descriptor=False):
    # TODO: refactor this function to run with numpy
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    if binary_descriptor:
        raise NotImplementedError("Binary descriptors are not implemented yet")
    else:
        d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
        d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

        eps = 1e-6
        return torch.sqrt((d1_sq.repeat(1, positive.size(0)) + torch.t(d2_sq.repeat(1, anchor.size(0)))
                           - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0)) + eps)


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


class DescriptorType(enum.Enum):
    SIFT = 1,
    SURF = 2,
    ORB = 3,
    FAST = 4,
    AKAZE = 5,


class MatchingStrategy(enum.Enum):
    FGINN_UNION = 0,
    FGINN_INTERSECT = 1,
    SNN = 2,  # second nearest neighbor
    BRUTE_FORCE = 3,


def decolorize(img):
    """
    colorize the image
    :param img: image to decolorize (BGR)
    :return: colorized image (GRAY)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _calculates_perspective_transform(homography_matrix, pts):
    """
    Calculates the perspective transform of the points

    :param homography_matrix: is the homography matrix
    :param pts: are the points
    :return: the perspective transform of the points
    """

    dst = cv2.perspectiveTransform(pts, homography_matrix)
    return dst


def _ransac_filtering(kps1, kps2, tentatives, threshold=1.0):
    """
    Filters the matches using RANSAC and calculates the homography matrix

    :param kps1: Keypoints of the first image
    :param kps2: Keypoints of the second image
    :param tentatives: are the matching pairs
    :param threshold: is the RANSAC threshold
    :return: the filtered matches and the homography matrix as tuple (matches, homography, mask)
    """

    good = []
    for i in range(len(tentatives)):
            good.append(cv2.DMatch(tentatives[i, 0], tentatives[i, 1], 1))
    src_pts = np.float32([kps1[m[0]].pt for m in tentatives]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m[1]].pt for m in tentatives]).reshape(-1, 1, 2)
    homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
    logging.debug("inliers found: %d" % (np.sum(mask)))
    if homography_matrix is None:
        print("homography matrix is None")
        raise ValueError("homography matrix is None")
    return good, homography_matrix, mask


def _get_fginn_indexes(dm, km, th=0.8):
    """

    :param dm: distance matrix of descriptors
    :param km: distance matrix of keypoints
    :param th: threshold
    :return:
    """
    vals, idx_in_2 = np.min(dm, dim=1)
    mask_1 = km <= 10.
    mask_2 = mask_1[idx_in_2, :]
    dm[mask_2] = 100000  # set to a large value
    vals_2nd, idx_in_taget_2nd = np.min(dm, dim=1)
    ratio = vals / vals_2nd
    mask = ratio <= th
    idx_in_1 = np.arange(0, idx_in_2.size[0])[mask]
    idx_in_2 = idx_in_2[mask]
    ml = np.cat([idx_in_1.reshape(-1, 1), idx_in_2.reshape(-1, 1)]).reshape(-1, 1)
    return ml


def _match_sym_fginn_intersect(desc_1, desc_2, kp_1, kp_2, th=0.8, desc_is_binary=False):
    """
    Matches the descriptors using the Symmetric FGINN intersection algorithm,
    described in D. Mishkin, J. Matas, M. Perdoch. MODS: Fast and Robust Method for Two-View Matching. CVIU 2015
    :param desc_1: descriptors of the first image
    :param desc_2: descriptors of the second image
    :param kp_1: keypoints of the first image
    :param kp_2: keypoints of the second image
    :param th: threshold for the ratio of the distances
    :param desc_is_binary: if the descriptors are binary which is needed to know for the distance matrix

    :return: the intersection matches as list of tuples (idx_1, idx_2)
    """

    # store keypoint as xy tuples and convert to numpy array
    xy1 = np.concatenate([np.array(p.pt).reshape(1, 2) for p in kp_1], axis=0)
    xy2 = np.concatenate([np.array(p.pt).reshape(1, 2) for p in kp_2], axis=0)

    # compute the distance matrix
    dm = distance_matrix(desc_1.astype(np.float32), desc_2.astype(np.float32))

    # compute the distance matrix of keypoints
    # ATTENTION: Keypoint's have to be in the same image
    km1 = distance_matrix(xy2.astype(np.float32), xy2.astype(np.float32))
    km2 = distance_matrix(xy1.astype(np.float32), xy1.astype(np.float32))
    # compute the indexes of the matches
    ml_1 = _get_fginn_indexes(dm, km1, th)
    ml_2 = _get_fginn_indexes(dm, km2, th)

    # compute the intersection of the indexes
    out_m = []
    for i in range(len(ml_1)):
        i1, i2 = ml_1[i]
        mask = ml_2[:, 0] == i1
        row = ml_2[mask]
        if len(row) > 0:
            i1l, i2l = row[0]
            if (i1 == i1l) and (i2 == i2l):
                out_m.append(ml_1[i].reshape(-1, 1))  # append the match

            if len(out_m) > 0:
                return np.concatenate(out_m, axis=0).astype(np.int32)  # return the matches as integers
            return np.zeros((2, 0), dtype=np.int32)  # return an empty array if no match was found

    return np.concatenate([ml_1, ml_2], axis=0)


def draw_matches(img1, img2, kp1, kp2,pts, dst, matches, color=(0, 255, 0), write_to_file=None):
    """

    :param img1: reference image as np array
    :param img2: target image as np array
    :param kp1: keypoints of first image
    :param kp2: keypoints of second image
    :param pts: given points in the reference image
    :param dst: given points in the target image
    :param matches: the matches
    :param color: color of the matches
    :param write_to_file: path of image or None if no image should be written
    :return:
    """
    draw_params = dict(matchColor=color,
                       singlePointColor=None,
                       matchesMask=matches,
                       flags=2)
    img3 = cv.polylines(img1, [pts],True, color=(255,0,0), thickness=2)
    img3 = cv.polylines(img2, [dst],True, color=(0,255,0), thickness=2)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)  # draw the matches
    if write_to_file is not None:
        logging.debug("writing matches to file: %s" % write_to_file)
        cv.imwrite(write_to_file, img3)
    # check if the image is gray or color
    if img3.shape[3] == 1:
        plt.imshow(img3, 'gray')
    else:
        plt.imread(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
    plt.show()
    return


class HomographyProvider:
    """
    A class which provides homography matrices for a given board orientation
    """

    def __init__(self, descriptor_type=DescriptorType.SIFT, matching=MatchingStrategy.SNN, display_result=False):

        self.matcher = None
        self.detector = None
        self.matching_strategy = None
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
        elif descriptor_type == DescriptorType.AKAZE:
            self.detector = cv.AKAZE_create()
        else:
            raise Exception("Unknown descriptor type")

    def _init_matching(self, strategy: MatchingStrategy):
        """
        Initializes the matcher based on the matching type
        :param matching:
        :return:
        """
        if strategy == MatchingStrategy.BRUTE_FORCE:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif strategy == MatchingStrategy.FGINN_INTERSECT:
            self.matching_strategy = _match_sym_fginn_intersect

    def _get_key_points(self, img_1, img_2, decolorized=False):
        """
        Returns the key points and descriptors
        :param img_1: image as np array
        :param img_2: image as np array
        :return: key points and descriptors using selected detector
        """
        if self.detector is None:
            raise Exception("Detector not initialized")
        if decolorized:
            img_1 = decolorize(img_1)
            img_2 = decolorize(img_2)

        kp1, des1 = self.detector.detectAndCompute(img_1, None)
        kp2, des2 = self.detector.detectAndCompute(img_2, None)
        return kp1, kp2, des1, des2

    def _get_homography(self, ref_image, tar_image, decolorized=False, ransac_threshold=2.0, should_draw=False,
                        write_to_file=None):
        """
        Calculates the homography matrix between two images using classical image matching pipeline
        (See Image Matching Across Wide Baselines: From Paper to Practice 2021)
        :param ref_image: reference image
        :param tar_image: target image
        :param decolorized: if the images should be decolorized (convert to grayscale)
        :param ransac_threshold: threshold for ransac epi-polar line
        :param should_draw: if the matches should be drawn
        :param write_to_file: path to image or None if no image should be written

        :return: the
        """
        if self.matcher is None:
            raise Exception("Matcher not initialized")
        # detect and desctibe the keypoints
        kp1, kp2, des1, des2 = self._get_key_points(ref_image, tar_image, decolorized)
        # match the keypoints using the selected strategy
        matches = self.matching_strategy(des1, des2, kp1, kp2)
        if len(matches) == 0:
            print("No matches found")
            return None
        # ransac filter the matches
        # TODO: implement other ransac strategies (e.g. MSAC)
        good, homography_matrix, mask = _ransac_filtering(kp1, kp2, matches, threshold=ransac_threshold)
        if len(good) == 0:
            print("No good matches found")
            return None

        # get corners of the image
        h, w, ch = ref_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # get the corners of the transformed image  (using the homography matrix)
        dst = _calculates_perspective_transform(homography_matrix, pts)
        # draw the matches
        if should_draw:
            draw_matches(ref_image, kp1, tar_image, kp2, good, write_to_file)

        return BoardOrientation(homography_matrix, dst, ref_image.shape[:2])




