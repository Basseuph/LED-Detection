import logging
import queue
import sys

import ffmpeg
import cv2
import logging as log

video_format = "flv"
server_url = "rtmp://localhost:8080"


class VideoStream:
    """
    VideoStream class
    for creating an rtmp stream out of opencv frames
    """

    def __init__(self, url, publish_stream, logger: log.Logger = logging.root):

        self.url = url
        self.publish_stream = publish_stream
        self.process = None
        self.logger = logger

    def write(self, frame):
        """
        Writes the frame to the ffmpeg process.
        !If the flag publish_stream is set, this method will return without doing anything!
        :param frame: is an opencv frame type: numpy.ndarray
        :return:
        """

        if not self.publish_stream:
            return

        if not self.process:
            self.start_streaming(frame.shape[1], frame.shape[0])
        try:
            self.process.stdin.write(frame.tobytes())
        except IOError as e:
            self.logger.error("Error writing to ffmpeg process: {}".format(e))
            self.stop_streaming()
        except BrokenPipeError as e:
            self.logger.error("Error writing to ffmpeg process: {}".format(e))
            self.stop_streaming()

    def start_streaming(self, width, height, fps=30):
        """
        Starts the ffmpeg process to stream the video and stores it in self.process
        :param width: is the input and output width of the video type: int
        :param height: is the input and output height of the video type: int
        :param fps:
        :return:
        """
        self.process = (
            ffmpeg
                .input('pipe:', format='rawvideo', codec="rawvideo", pix_fmt='bgr24', s='{}x{}'.format(width, height))
                .output(
                self.url,
                # codec = "copy", # use same codecs of the original video
                listen=1,  # enables HTTP server
                pix_fmt="yuv420p",
                preset="ultrafast",
                f=video_format,

            )
                .global_args('-re')
                .overwrite_output()
                .run_async(pipe_stdin=True)
        )
        self.logger.info("Start streaming to {}".format(self.url))

    def stop_streaming(self):
        """
        Stops the ffmpeg process
        :return:
        """
        if self.process:
            self.process.kill()
            self.process = None
            self.logger.info("Stopped streaming to {}".format(self.url))
