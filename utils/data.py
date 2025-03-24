import sys
import os, re
import logging
import torch
import numpy as np
#sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

class Checkerboard(object):
    def __init__(self, N=1000, x_min=-4, x_max=4, y_min=-4, y_max=4, length=4):
        """
        Args:
            N: Number of points to sample
            x_min and x_max: min and max values over x axis
            y_min and y_max: min and max values over y axis
            length: length of checkboard pattern
        """
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.length = length
        # Checkerboard pattern
        self.checkerboard_pattern = np.indices((self.length, self.length)).sum(axis=0) % 2
        self.sampled_points = self._sample_checkerboard_data()

    def _sample_checkerboard_data(self):
        """
        Return a ndarray of sampled points that follows a checkerboard pattern
        """
        sampled_points = []
        # Sample points in regions where checkerboard pattern is 1
        while len(sampled_points) < self.N:
            # Randomly sample a point within the x and y range
            x_sample = np.random.uniform(self.x_min, self.x_max)
            y_sample = np.random.uniform(self.y_min, self.y_max)
            
            # Determine the closest grid index
            i = int((x_sample - self.x_min) / (self.x_max - self.x_min) * self.length)
            j = int((y_sample - self.y_min) / (self.y_max - self.y_min) * self.length)
            
            # Check if the sampled point is in a region where checkerboard == 1
            if self.checkerboard_pattern[j, i] == 1:
                sampled_points.append((x_sample, y_sample))

        # Convert to NumPy array for easier plotting
        sampled_points = np.array(sampled_points)
        logging.info(f'Sampled points shape:{sampled_points.shape}')
        return sampled_points