import sys

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


class ImagePreprocessor():
    def __init__(self,   size=(480, 640, 3)):
        self.size = size
        self.counter = 0
        self.images = []

    def add_img(self, img):
        """
        adds an image to the batch
        :param img: the image to be added
        :return: None
        """
        self.images.append(img)



    def preprocess(self):
        """
        preprocesses a batch of images
        :param image_batch: a batch of images
        :return: a single prepocessed image
        """
        images = np.array(self.images)
        image = np.mean(images, axis=0).astype(np.uint8)
        # convert from RGB color-space to YCrCb
        return self.historgram_equalization(image)

    def historgram_equalization(self, img, use_clahe=True):
        """

        :param img:
        :return:
        """
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        # equalize the histogram of the brightness channel
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])
        else:
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        return equalized_img


if __name__ == '__main__':
    vid = cv2.VideoCapture("/Users/christophmeyer/Documents/uni/ws2022/se-project/tests/resources/output.avi")
    image_preprocessor = ImagePreprocessor()
    n = 0
    while True:
        ret, frame = vid.read()
        if frame is not None:
            image_preprocessor.add_img(frame)
            n += 1
            if n % 100 == 0:

                img = image_preprocessor.preprocess()
                cv2.imshow("preprocessed", img)
                cv2.imshow("original", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    vid.release()
                    cv2.destroyAllWindows()
                    sys.exit()

