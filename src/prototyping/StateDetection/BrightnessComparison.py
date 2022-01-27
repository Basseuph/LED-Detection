import cv2
import collections


def avg(hist: [[int]]) -> int:
    """
    Returns the average in a histogram, returned by cv2.calcHist.
    :param hist: The histogram.
    :return: The average of the given histogram.
    """
    s = 0
    e = 0
    for i in range(len(hist)):
        e += hist[i, 0]
        s += i * hist[i, 0]
    return int(s / e)


class BrightnessComparison:
    def __init__(self, deviation: int = 10):
        """
        :param deviation: the deviation of the average on value.
        """
        self._last_brightness = -1
        self._on_values = collections.deque(maxlen=20)
        self._deviation = deviation

    def detect(self, img, window_name: str = None):
        """
        True - LED is powered on.
        False - LED is powered off.
        None - LED is in an undefined state.
        :param img: The BGR image of this LED.
        :param window_name: Set a name, to display a cv2 window with the given img in grayscale.
        :return: True if LED is powered on or None if undefined.
        """
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if window_name is not None:
            cv2.imshow(window_name, gray_img)

        # get the average brightness in the given image
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
        hist_avg = avg(hist)

        # if no known brightness values for on
        if len(self._on_values) == 0:
            if hist_avg in range(self._last_brightness - self._deviation,
                                 self._last_brightness + self._deviation) or self._last_brightness == -1:
                self._last_brightness = hist_avg
                return None
            elif hist_avg > self._last_brightness:
                self._on_values.append(hist_avg)
                return True
            else:
                self._on_values.append(self._last_brightness)
                return False
        else:
            on_avg = int(sum(self._on_values) / len(self._on_values))
            if hist_avg in range(on_avg - self._deviation, 256):
                self._on_values.append(hist_avg)
                return True
            return False
