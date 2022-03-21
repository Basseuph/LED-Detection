import cv2, queue, threading, time


# bufferless VideoCapture
# https://stackoverflow.com/a/54755738
class BufferlessVideoCapture:
    """
    A special VideoCapture which will always return the most recent frame instead of the next available, meaning
    that all frames except the most recent one are dropped.
    Opens a cv2 VideoCapture with the given name what can be the webcam id to be wrapped.
    """

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        # set camera settings
        self._set_camera_settings_HD()

        if not self.cap.isOpened():
            raise IOError("Couldn't open stream for device {}".format(name))


        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _set_camera_settings_HD(self):
        """
        Sets the camera settings to .
        :return:
        """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

