import time
import colorsys
import cv2

from StateDetection.BrightnessComparison import BrightnessComparison
from ColorDetection.HueComparison import Comparison
from ColorDetection import KMeans
from ColorDetection import DominantColor
from ColorDetection import Util
from StateDetection.BackgroundSubtractor import BackgroundSubtraction


class LedStateDetector:
    _counter = 0

    def __init__(self, bbox, name: str = None, colors: [str] = None):
        """
        Bounding box should be (left, top, right, bottom).
        Current LED state can be checked with is_on.
        Current LED color can be checked with color.
        The time since the last state change can be checked using passed_time.
        None is used as an undefined state for color, is_on and passed_time.
        :param bbox: the bounding box to define the LED's location.
        :param name: the name / identification for this LED.
        :param colors: all colors that should be checked for on this LED.
        """
        self._bbox = bbox
        self._brightness = BrightnessComparison()
        self._last_state_time = None
        self._id = LedStateDetector._counter
        LedStateDetector._counter += 1
        self._sub = BackgroundSubtraction()
        self._hue_comparison = Comparison(colors)

        if name is None:
            self.name = str(self._id)
        else:
            self.name = name
        self.is_on = None
        self.passed_time = None
        self.color = None

    def detect(self, image, imshow: bool = False):
        """
        Checks if the LED at the to be observed location is powered on in the given image.
        If the LED changed it's state, the color will be checked.
        Returns True if the LED has changed it's state i.e. from on to off.
        :param imshow: If set to True, an image with the given defined bbox will be displayed using cv2.imshow().
        :param image: The BGR image of the board that should be checked.
        :return: True if the led has changed it's state.
        """
        img = self.frame_cutout(image)
        sub_result = self._sub.detect(img, "Subtraction")

        if imshow:
            on = self._brightness.detect(img, self.name + " Gray")
        else:
            on = self._brightness.detect(img)

        change = on is not None and (self.is_on is None or on is not self.is_on)
        if change:
            print("Sub:", sub_result)
            self._state_change(on)
            comparison = self._hue_comparison.color_detection(img, on)
            if self.is_on:
                rgb = KMeans.k_means(img)
                hsv = colorsys.rgb_to_hsv(rgb[0] / float(255), rgb[1] / float(255), rgb[2] / float(255))
                k_mean = Util.get_color(int(hsv[0] * 180))
                d = DominantColor.get_dominant_color_value(img)
                dominant = Util.get_color(d)

                print(self.name, "KMean", k_mean)
                print(self.name, "HueComparison", comparison)
                print(self.name, "Dominant", dominant)
        elif self.is_on is None:
            self._hue_comparison.color_detection(img, self.is_on)
        return change

    def _state_change(self, state: bool) -> None:
        self.is_on = state
        if self._last_state_time is None:
            self._last_state_time = time.time()
        else:
            current = time.time()
            self.passed_time = current - self._last_state_time
            self._last_state_time = current

    def frame_cutout(self, img):
        """
        Returns a cutout of the image of the defined bbox for this LED.
        :param img: The image of the board.
        :return: A cutout of the given image using this LED's defined bbox.
        """
        return img[self._bbox[1]:self._bbox[3], self._bbox[0]:self._bbox[2], :]


if __name__ == '__main__':
    tests = {
        cv2.VideoCapture("resources/output.avi"): [
            # LedStateDetector((439, 340, 452, 349), name="Counter[0]", colors=["green", "yellow", "red"]),
            # LedStateDetector((440, 335, 447, 339), name="Counter[1]", colors=["green", "yellow", "red"]),
            # LedStateDetector((438, 327, 449, 334), name="Counter[2]", colors=["green", "yellow", "red"]),
            LedStateDetector((473, 392, 486, 410), name="Blinking", colors=["red", "green", "yellow", "cyan"]),
        ],
    }

    for video in tests:
        leds = tests.get(video)

        cv2.namedWindow('Raw', cv2.WINDOW_KEEPRATIO)
        frame_exists, frame = video.read()
        while frame_exists:
            for led in leds:
                if led.detect(frame, True):
                    if led.is_on:
                        print(led.name, "ON", "Time passed:", led.passed_time)
                    else:
                        print(led.name, "OFF", "Time passed:", led.passed_time)
            cv2.imshow("Raw", frame)
            cv2.resizeWindow("Raw", 500, 500)
            key = cv2.waitKey(1)
            if key == 27:
                break
            if key == 32:
                cv2.imwrite("frame.png", frame)
            frame_exists, frame = video.read()
        cv2.destroyAllWindows()
