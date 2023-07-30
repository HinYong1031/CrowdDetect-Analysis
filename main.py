import os
import csv
from detection import processVideo

if not os.path.exists('processed_data'):
	os.makedirs('processed_data')

movement_data_file = open('processed_data/movement_data.csv', 'w') 
crowd_data_file = open('processed_data/crowd_data.csv', 'w')


movement_data_writer = csv.writer(movement_data_file)
crowd_data_writer = csv.writer(crowd_data_file)


if os.path.getsize('processed_data/movement_data.csv') == 0:
	movement_data_writer.writerow(['Track ID', 'Entry Time', 'Exit Time', 'Movement Tracks'])
if os.path.getsize('processed_data/crowd_data.csv') == 0:
	crowd_data_writer.writerow(['Time', 'Human Count'])

model = "yolov8x.pt"
input_video_path = "BusStop3.MOV"
processVideo(input_video_path, model, movement_data_writer) 
movement_data_file.close()
crowd_data_file.close()