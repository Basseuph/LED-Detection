from time import time


class MatrixCache:
    def __init__(self, matrix, corners):
        self.matrix = matrix
        self.corners = corners
        self.timestamp = time()

    def check_if_outdated(self):
        return time() - self.timestamp >= 3
