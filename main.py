# main.py

import argparse
from lane_detector import LaneDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane Detection Service")
    parser.add_argument("source", type=str, help="Path to video file or camera index (0 for default camera)")
    parser.add_argument("algorithm", type=str, choices=['hough', 'color', 'lsd'], help="Algorithm to use for lane detection")
    
    args = parser.parse_args()

    lane_detector = LaneDetector(args.source, args.algorithm)
    lane_detector.process_video()
