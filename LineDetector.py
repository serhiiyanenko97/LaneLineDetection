import numpy as np
import cv2
from screeninfo import get_monitors
import argparse

class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def region_selection(image):
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    @staticmethod
    def preprocess_image(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        roi = ImageProcessor.region_selection(edges)
        return roi


class LineDetector:
    def __init__(self):
        pass

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
    def lsd_line_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(gray)[0]  # Get detected lines

        if lines is None or len(lines) == 0:
            return []  # Return an empty list if no lines are detected

        # Ensure lines are in the correct format: list of tuples
        formatted_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Get the points from the line
            formatted_lines.append(((int(x1), int(y1)), (int(x2), int(y2))))  # Append as tuple of tuples


        return formatted_lines
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

class LaneDetector:
    def __init__(self, source, algorithm):
        self.source = source
        self.algorithm = algorithm
        self.image_processor = ImageProcessor()
        self.line_detector = LineDetector()

    def color_line_detection(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 100, 200])  # Adjust based on your needs
        upper_white = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        if np.sum(mask) == 0:
            return None  # Return None if no lanes detected in the mask
        return mask  # Return the mask if lanes were detected

    def process_frame(self, frame):
        if self.algorithm in ['hough', 'color', 'lsd']:  # Check for valid algorithms
            processed_frame = self.image_processor.preprocess_image(frame)

        if self.algorithm == 'hough':
            hough_lines = self.line_detector.hough_transform(processed_frame)
            if hough_lines is not None and len(hough_lines) > 0:  # Checking for non-empty lines
                return self.line_detector.draw_lane_lines(frame, self.line_detector.lane_lines(frame, hough_lines))
            return frame  # Return original frame if no lines detected
        
        elif self.algorithm == 'color':
            color_mask = self.color_line_detection(frame)
            if color_mask is None or np.sum(color_mask) == 0:  # No detected edges
                return frame  # Return original image if no edges found

            hough_lines = self.line_detector.hough_transform(color_mask)
            if hough_lines is not None and len(hough_lines) > 0:
                return self.line_detector.draw_lane_lines(frame, self.line_detector.lane_lines(frame, hough_lines))
            return frame  # Return original image if no lines detected

        elif self.algorithm == 'lsd':
            lines = self.line_detector.lsd_line_detection(frame)
            return self.line_detector.draw_lane_lines(frame, lines) if lines else frame  # Handle no detected lines
        
        raise ValueError("Invalid algorithm selected.")

    def process_video(self):
        monitor = get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Error: Could not open video or camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            resized_frame = cv2.resize(processed_frame, (screen_width, screen_height))
            cv2.imshow('Lane Detection', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection Service")
    parser.add_argument("source", type=str, help="Path to video file or camera index (0 for default camera)")
    parser.add_argument("algorithm", type=str, choices=['hough', 'color', 'lsd'], help="Algorithm to use for lane detection")
    
    args = parser.parse_args()

    lane_detector = LaneDetector(args.source, args.algorithm)
    lane_detector.process_video()
