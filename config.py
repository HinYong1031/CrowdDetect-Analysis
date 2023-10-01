import os

# Object detection configuration
DETECTION_MODEL = "yolov8l.pt"
INPUT_VIDEO_PATH = "..\ch06_20230703115412.mp4"

# Directory for processed data
PROCESSED_DATA_DIR = 'processed_data'

# File paths for movement and crowd data
MOVEMENT_DATA_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, 'movement_data.csv')
CROWD_DATA_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, 'crowd_data.csv')

# detection.py configuration
NEW_DESIRED_FRAME = 10
OUTPUT_PATH = 'output_videos'
VIDEO_SAVE_PATH = os.path.join(OUTPUT_PATH, "output.mp4")
AREA1 = [(635, 455), (662, 443), (862, 540), (790, 540)]
AREA2 = [(604, 467), (730, 540), (680, 540), (576, 478)]
SAVE_VIDEO = False