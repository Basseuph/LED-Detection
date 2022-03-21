import cv2


class BackgroundSubtraction:
    def __init__(self, history: int = 1):
        self._sub = cv2.createBackgroundSubtractorMOG2(history=history, detectShadows=False)
        self._last_value = 0

        self.counter = 0
        self.change = False

    def detect(self, frame, window_name: str = None):
        """
        :param frame: BGR frame.
        :param window_name: debugging.
        :return: True if a change has been detected.
        """
        h, w = len(frame), len(frame[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        mask = self._sub.apply(blur)
        if window_name is not None:
            cv2.imshow(window_name, mask)

        # np_array will use uint8 in sum function
        py_mask = mask.tolist()
        s = sum([sum(sub_list) for sub_list in py_mask])
        max_value = h * w * 255
        # the amount of changing pixels needed to trigger a 'change'
        change_percentage = 0.30
        change = s >= change_percentage * max_value
        if change and not self.change:
            # print("py_mask", py_mask)
            print("change", self.counter)
            self.counter += 1
            return True
        self.change = change
        return False
