import collections
import time

import cv2

# blob detection params
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.filterByConvexity = False
params.filterByArea = False
params.filterByInertia = True
params.filterByCircularity = True
params.minInertiaRatio = 0.1
params.minCircularity = 0.1
_detector = cv2.SimpleBlobDetector_create(params)


def avg(l) -> int:
    s = 0
    e = 0
    for i in range(len(l)):
        e += l[i, 0]
        s += i * l[i, 0]
    return int(s / e)


class LedStateDetector:
    _counter = 0

    def __init__(self, bbox):
        """
        Bounding box should be (left, top, right, bottom).
        :param bbox: the bounding box to define the LED's location.
        """
        self._bbox = bbox
        self._last_brightness = -1
        self._on_values = collections.deque(maxlen=20)
        self._last_state_time = None

        self.id = LedStateDetector._counter
        LedStateDetector._counter += 1
        self.is_on = False
        self.passed_time = None

    def detect(self, img, imshow: bool = False):
        """
        Checks if the LED at the to be observed location is powered on in the given image.
        Returns True if the LED has changed it's state i.e. from on to off.
        The current LED's state is stored in is_on.
        :param imshow: if set to True, an image with the given defined bbox will be displayed using cv2.imshow().
        :param img: the image that should be checked.
        :return: True if the led has changed it's state.
        """
        img = img[self._bbox[1]:self._bbox[3], self._bbox[0]:self._bbox[2], :]

        # Detection using brightness changes seems to be more reliable and accurate (needs further testing)
        # on = self._detect_using_blob(img, imshow)

        on = self._detect_using_brightness(img, imshow)
        if on and not self.is_on:
            # LED turned on
            self._state_change(on)
            return True
        elif not on and self.is_on:
            # LED turned off
            self._state_change(on)
            return True
        return False

    def _state_change(self, state: bool):
        self.is_on = state
        if self._last_state_time is None:
            self._last_state_time = time.time()
        else:
            current = time.time()
            self.passed_time = current - self._last_state_time
            self._last_state_time = current

    def _detect_using_brightness(self, img, imshow: bool = False):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        if imshow:
            cv2.imshow(str(self.id) + "Gray", gray_img)
        hist_avg = avg(hist)
        if len(self._on_values) == 0:
            if hist_avg in range(self._last_brightness - 10, self._last_brightness + 10) or self._last_brightness == -1:
                self._last_brightness = hist_avg
            elif hist_avg > self._last_brightness:
                self._on_values.append(hist_avg)
            else:
                self._on_values.append(self._last_brightness)
        else:
            on_avg = int(sum(self._on_values) / len(self._on_values))
            if hist_avg in range(on_avg - 10, 256):
                self._on_values.append(hist_avg)
                return True
            return False

    def _detect_using_blob(self, img, imshow: bool = False):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(gray_img, 215, 255, cv2.THRESH_BINARY)
        thresh_img = cv2.erode(thresh_img, None, iterations=2)
        thresh_img = cv2.dilate(thresh_img, None, iterations=2)
        if imshow:
            cv2.imshow(str(self.id) + "Threshold", thresh_img)
        keypoints = _detector.detect(thresh_img)
        return len(keypoints) > 0


if __name__ == '__main__':
    led1 = LedStateDetector((980, 620, 1010, 660))
    led2 = LedStateDetector((980, 660, 1010, 700))
    vid = cv2.VideoCapture("resources/piOnOff.mp4")

    leds = [led1, led2]
    frame_exists, frame = vid.read()
    while frame_exists:
        for led in leds:
            if led.detect(frame, True):
                if led.is_on:
                    print("LED", led.id, "ON", "Time passed:", led.passed_time)
                else:
                    print("LED", led.id, "OFF", "Time passed:", led.passed_time)
        cv2.imshow("Raw", frame)
        if cv2.waitKey(10) == 27:
            break
        frame_exists, frame = vid.read()
    cv2.destroyAllWindows()
