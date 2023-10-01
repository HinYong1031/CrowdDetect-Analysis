import os
import csv
from detection import processVideo
import threading
from optical import opticalFlow
from config import DETECTION_MODEL, INPUT_VIDEO_PATH, PROCESSED_DATA_DIR, MOVEMENT_DATA_FILE_PATH, CROWD_DATA_FILE_PATH


# run object detection
def object_detection():
	if not os.path.exists(PROCESSED_DATA_DIR):
		os.makedirs(PROCESSED_DATA_DIR)

	movement_data_file = open(MOVEMENT_DATA_FILE_PATH, 'a') 
	crowd_data_file = open(CROWD_DATA_FILE_PATH, 'w')
	movement_data_file.truncate(0)	# clear file

	movement_data_writer = csv.writer(movement_data_file)
	crowd_data_writer = csv.writer(crowd_data_file)

	if os.path.getsize(MOVEMENT_DATA_FILE_PATH) == 0:
		movement_data_writer.writerow(['Track ID', 'Entry Time (s)', 'Exit Time (s)', 'Movement Tracks'])
	if os.path.getsize(CROWD_DATA_FILE_PATH) == 0:
		crowd_data_writer.writerow(['Time', 'Human Count'])

	model = DETECTION_MODEL
	input_video_path = INPUT_VIDEO_PATH
	processVideo(input_video_path, model, movement_data_writer, movement_data_file, crowd_data_writer)

	movement_data_file.close()
	crowd_data_file.close()

# run optical flow
def optical_flow():
	opticalFlow()

# Create threads for each project
detection_thread = threading.Thread(target=object_detection)
optical_thread = threading.Thread(target=optical_flow)

# Start the threads
detection_thread.start()
optical_thread.start()

# Wait for both threads to finish
detection_thread.join()
optical_thread.join()