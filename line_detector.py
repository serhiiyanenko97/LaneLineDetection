# line_detector.py

import numpy as np
import cv2

class LineDetector:
    @staticmethod
    def hough_transform(image):
        rho = 1
        theta = np.pi / 180
        threshold = 50  # Minimum votes to consider a line
        min_line_length = 50  # Minimum length of a line
        max_line_gap = 30  # Maximum allowed gap between points on the same line
        lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        return lines if lines is not None else []

    @staticmethod
    def average_slope_intercept(lines):
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:  # Skip vertical lines
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:  # Left lane
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:  # Right lane
                    right_lines.append((slope, intercept))
                    right_weights.append(length)

        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
        return left_lane, right_lane

    @staticmethod
    def pixel_points(y1, y2, line):
        if line is None:
            return None
        
        slope, intercept = line
        
        if np.isclose(slope, 0):  # Handle horizontal lines
            x1 = x2 = int(intercept)
        elif np.isinf(slope):  # Handle vertical lines
            x1 = x2 = int(intercept)
        else:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

        return ((x1, y1), (x2, y2))

    @staticmethod
    def lane_lines(image, lines):
        left_lane, right_lane = LineDetector.average_slope_intercept(lines)
        y1 = image.shape[0]  # bottom of the image
        y2 = int(y1 * 0.6)   # a little above the center of the image
        left_line = LineDetector.pixel_points(y1, y2, left_lane)
        right_line = LineDetector.pixel_points(y1, y2, right_lane)
        return left_line, right_line

    @staticmethod
    def draw_lane_lines(image, lines, color=(0, 255, 0), thickness=5):
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, line[0], line[1], color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 0.6, 0.0)
