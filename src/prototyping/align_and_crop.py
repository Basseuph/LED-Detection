from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights
    Args:
        corners: list
    Returns:
        destination_corners: list
        height: int
        width: int
    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

    #print('\nThe destination points are: \n')
    #for index, c in enumerate(destination_corners):
    #    character = chr(65 + index) + "'"
    #    print(character, ':', c)

    #print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w


def unwarp(img, src, dst):
    """
    Args:
        img: np.array
        src: list
        dst: list
    Returns:
        un_warped: np.array
    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    #print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    # plot

    #f, (ax1, ax2) = plt.subplots(1, 2)
    #ax1.imshow(img)
    #x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
    #y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
    #ax1.plot(x, y, color='yellow', linewidth=3)
    #ax1.set_ylim([h, 0])
    #ax1.set_xlim([0, w])
    #ax1.set_title('Targeted Area in Original Image')
    #ax2.imshow(un_warped)
    #ax2.set_title('Unwarped Image')
    #plt.show()
    return un_warped, H


def align_and_crop(img, corners):
    destination_points, h, w = get_destination_points(corners)
    un_warped, H = unwarp(img, np.float32(corners), destination_points)

    cropped = un_warped[0:h, 0:w]
    cv2.imwrite("./result.jpg", cropped)

