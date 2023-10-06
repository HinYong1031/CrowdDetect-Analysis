from ultralytics import YOLO 
import cv2
import cvzone
import numpy as np
import os
import imutils
import datetime
import pandas as pd
from config import NEW_DESIRED_FRAME, OUTPUT_PATH, VIDEO_SAVE_PATH, AREA1, AREA2, SAVE_VIDEO

#YoloV8 official model label data, this project only uses 'person'
# classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#'scissors', 'teddy bear', 'hair drier', 'toothbrush']

IS_CAM = False

# Save dict of tracked objects into a csv file (将跟踪对象的字典保存到csv文件中)
def _record_movement_data(movement_data_writer,tracked_objects,file):
    for track_id in tracked_objects.keys():
        entry_time = tracked_objects[track_id]['Entry Time']
        exit_time = tracked_objects[track_id]['Exit Time']
        movement_tracks = tracked_objects[track_id]['Movement Tracks']
        movement_data_writer.writerow([track_id, entry_time, exit_time, movement_tracks])
        
    # Flush the file buffer to ensure data is written to file (刷新文件缓冲区以确保数据写入文件)
    file.flush()
    movement_data_writer.writerow([])
    file.flush()
    df = pd.read_csv('processed_data/movement_data.csv')
    unique_df = df.dropna(subset=['Track ID'])
    # Drop duplicate rows based on the 'ID' column, keeping only the last occurrence (most recent row)
    unique_df = df.drop_duplicates(subset=['Track ID'], keep='last')
    unique_df.to_csv('processed_data/movement_data.csv', index=False)

def _record_crowd_data(crowd_data_writer,time,human_count):
    data = [time, human_count]
    crowd_data_writer.writerow(data)


def processVideo(inputPath,model,movement_data_writer,movement_data_file,crowd_data_writer):

    #Load YOLO model file (加载YOLO模型文件)
    model=YOLO(model)
    model.to('cuda')
    #Read the video from inputPath (从inputPath读入视频)
    cap = cv2.VideoCapture(inputPath)

    #Get the frame rate of the video (获取视频的帧率)
    fps = cap.get(cv2.CAP_PROP_FPS)

    DATA_RECORD_FRAME = fps / NEW_DESIRED_FRAME

    #Get the size of the video (获取视频的大小)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2 )
    size = (w, h)
    
    #Initialize video writing
    if SAVE_VIDEO:
        output_video = cv2.VideoWriter()
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        #Create output video path (创建输出视频路径)
        output_video.open(VIDEO_SAVE_PATH, codec, fps, size, True)

    # Initialization variables (初始化变量)
    tracked_objects = {}
    frame_count = 0
    people_entering, entering = {}, set()
    people_exiting, exiting = {}, set()
    totalCount = []

    #Read and process each frame of the picture(对每一帧图片进行读取和处理)
    while cap.isOpened():
        #Read frame from video
        ret, img = cap.read()

        #Exit if no picture is read (如果没有读取到图片则退出)
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1

        # Skip frames according to given rate (根据给定的速率跳过帧)
        if frame_count % int(DATA_RECORD_FRAME) == 0:
            #Resize Frame to half of its size for faster display in imshow (将帧的大小调整为其一半，以便在imshow中更快地显示)
            img = imutils.resize(img, width=w, height=h)

            # Get current time
            current_datetime = datetime.datetime.now()

            # Run detection algorithm
            if IS_CAM:
                record_time = current_datetime
            else:
                record_time = frame_count

            #Infer each frame of the picture
            results = model.track(img, tracker='botsort.yaml', persist=True, verbose=False)
            
            for r in results:
                boxes = r.boxes
                # Handle the case when no objects are detected
                # skip to next frame (处理未检测到对象的情况，跳到下一帧)
                if r.boxes.id is None:
                    continue
                for box in boxes:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                    id = int(box.id)
                    centre_x = int((x1 + x2) / 2)
                    centre_y = int((y1 + y2) / 2)
                    if box.cls == 0:
                        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)
                        #在跟踪框上写入ID
                        cvzone.putTextRect(img,f'{int(id)}',(max(-10,x1),max(-40,y1)),scale=1,thickness=1,
                                        offset=2)

                        # Check if object is entering
                        enterTest = cv2.pointPolygonTest(np.array(AREA1,np.int32),((x1,y2)),False)
                        if enterTest >= 0:
                            people_entering[id]=(x1,y2)
                        if id in people_entering:
                            confirmIn = cv2.pointPolygonTest(np.array(AREA2,np.int32),((x1,y2)),False)
                            if confirmIn >= 0:
                                entering.add(id)
                        
                        # Check if object is exiting
                        leaveTest = cv2.pointPolygonTest(np.array(AREA2,np.int32),((x1,y2)),False)
                        if leaveTest >= 0:
                            people_exiting[id]=(x1,y2)
                        if id in people_exiting:
                            confirmOut = cv2.pointPolygonTest(np.array(AREA1,np.int32),((x1,y2)),False)
                            if confirmOut >= 0:
                                exiting.add(id)

                        # if we haven't seen a particular object ID before in entering, register it in a list 
                        if id in entering and id not in totalCount:
                            totalCount.append(id)

                        # Object is being tracked for the first time
                        if id not in tracked_objects and id in entering:
                            tracked_objects[id] = {
                                'Track ID': id,
                                'Entry Time': frame_count / fps,
                                'Exit Time': None,
                                'Movement Tracks': [[centre_x, centre_y]],
                            }
                        # Object is still being tracked
                        else:
                            # Existing id, update movement tracks
                            if id in tracked_objects:
                                movement_tracks = tracked_objects[id]['Movement Tracks']
                                movement_tracks.append([centre_x, centre_y])

            # Handle objects that are not detected in the current frame
            for Id in list(tracked_objects.keys()):
                if tracked_objects[Id]['Exit Time'] is None and id in exiting:
                    tracked_objects[Id]['Exit Time'] = frame_count / fps

            # Draw the area of interest (画出感兴趣的区域)
            cv2.polylines(img,[np.array(AREA1,np.int32)],True,(255,0,0),2)
            cv2.putText(img,str('1'),(627,430),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0,0,0),1)
            cv2.polylines(img,[np.array(AREA2,np.int32)],True,(255,0,0),2)
            cv2.putText(img,str('2'),(587,465),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0,0,0),1)

            count_txt = len(totalCount)
            count_leave = len(exiting)
            total = count_txt - count_leave
            cv2.putText(img,"Entering: " + str(count_txt),(520,30),cv2.FONT_HERSHEY_SIMPLEX,(1),(0,0,0),2)
            cv2.putText(img,"Leaving: " + str(count_leave),(520,80),cv2.FONT_HERSHEY_SIMPLEX,(1),(0,0,0),2)

            _record_movement_data(movement_data_writer,tracked_objects,movement_data_file)
            _record_crowd_data(crowd_data_writer,record_time,total)

            # Write the processed image to the video (将处理后的图像写入视频)   
            if SAVE_VIDEO:
                output_video.write(img)

            # Show the processed image (显示处理后的图像)
            cv2.imshow("Processed Output", img)

            # Press q to exit (按q键退出)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # output_frames['detection'] = img

    cap.release() # Close the video file (关闭视频文件)
    cv2.destroyAllWindows() # Close the window (关闭窗口)
    if SAVE_VIDEO:
        output_video.release() # Close video writing (关闭视频写入)