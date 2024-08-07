# lane_detector.py

import cv2
import numpy as np
from line_detector import LineDetector
from image_processor import ImageProcessor
from screeninfo import get_monitors

class LaneDetector:
    def __init__(self, source, algorithm):
        self.source = source
        self.algorithm = algorithm
        self.image_processor = ImageProcessor()  # Initialize the ImageProcessor instance
        self.line_detector = LineDetector()  # Initialize the LineDetector instance

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
            if hough_lines is not None and len(hough_lines) > 0:
                return self.line_detector.draw_lane_lines(frame, self.line_detector.lane_lines(frame, hough_lines))
            return frame  # Return original frame if no lines detected
    
        elif self.algorithm == 'color':
            color_mask = self.color_line_detection(frame)
            if color_mask is None or np.sum(color_mask) == 0:  # No detected edges
                return frame  # Return original image if no edges found

            # Perform Hough Transform on the color mask
            hough_lines = self.line_detector.hough_transform(color_mask)
            if hough_lines is not None and len(hough_lines) > 0:
                return self.line_detector.draw_lane_lines(frame, self.line_detector.lane_lines(frame, hough_lines))
            return frame  # Return original image if no lines detected

        elif self.algorithm == 'lsd':
            lines = self.line_detector.lsd_line_detection(frame)
            return self.line_detector.draw_lane_lines(frame, lines) if lines else frame  # Handle no detected lines
        
        raise ValueError("Invalid algorithm selected.")

    def process_video(self):
        monitor = get_monitors()[0]  # Get the primary monitor resolution
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

            # Get the original dimensions of the input frame
            original_height, original_width = frame.shape[:2]

            # Calculate the scaling factor to maintain aspect ratio
            scaling_factor = min(screen_width / original_width, screen_height / original_height)

            # Resize processed frame
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)

            resized_frame = cv2.resize(processed_frame, (new_width, new_height))

            # Display the resized frame
            cv2.imshow('Lane Detection', resized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

