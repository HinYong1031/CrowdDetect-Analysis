from ultralytics import YOLO 
import math
import cv2
import cvzone
import numpy as np
import os
import imutils
from colors import RGB_COLORS
import datetime
import pandas as pd

#YoloV8 official model label data, this project only uses 'person'
#YoloV8官方模型标签数据，本次项目只使用了'person'
# classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#'scissors', 'teddy bear', 'hair drier', 'toothbrush']

IS_CAM = False
# area1 = [(423,580),(581,461),(620,465),(463,590)] # 1
# area2 = [(487,599),(635,471),(675,475),(530,610)] # 2
# area1 = [(284,365),(385,300),(416,305),(315,374)] # 1
# area2 = [(330,383),(428,312),(458,317),(367,394)] # 2
area1 = [(3,125),(27,124),(45,248),(17,248)] # 1
area2 = [(160,280),(430,240),(450,266),(180,310)] # 2
area3 = [(604,467),(730,540),(680,540),(576,478)] # 3
area4 = [(635,455),(662,443),(862,540),(790,540)] # 4


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

def heatmap(img, global_imgArray):
    # Heatmap array preprocessing (热力图数组预处理)
    global_imgArray_norm = (global_imgArray - global_imgArray.min()) / (global_imgArray.max() - global_imgArray.min()) * 255
    global_imgArray_norm = global_imgArray_norm.astype(np.uint8)
    # Apply Gaussian blur to remove noise (应用高斯模糊以去除噪声)
    global_imgArray_norm = cv2.GaussianBlur(global_imgArray_norm, (9, 9), 0)
    heatmap_img = cv2.applyColorMap(global_imgArray_norm, cv2.COLORMAP_SPRING)
    # Overlay heatmap on original image with 50% transparency (叠加热力图到原图，透明度50%)
    super_imposed_img = cv2.addWeighted(img, 0.5, heatmap_img, 0.5, 0)
    return super_imposed_img


def processVideo(inputPath,model,movement_data_writer,movement_data_file,crowd_data_writer):
    #Load YOLO model file (加载YOLO模型文件)
    model=YOLO(model)
    model.to('cuda')
    #Read the video from inputPath (从inputPath读入视频)
    cap = cv2.VideoCapture(inputPath)

    #Get the frame rate of the video (获取视频的帧率)
    fps = cap.get(cv2.CAP_PROP_FPS)
    #Get the size of the video (获取视频的大小)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2 )
    size = (w, h)
    
    #Initialize video writing
    output_video = cv2.VideoWriter()
    outputPath = 'output_videos'
    os.makedirs(outputPath, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    #Create output video path (创建输出视频路径)
    video_save_path = os.path.join(outputPath,"output.mp4")
    output_video.open(video_save_path, codec, fps, size, True)

    global_imgArray = None
    global_imgArray = np.ones([int(h), int(w)], dtype=np.uint32)

    # Initialize tracked_objects dictionary
    tracked_objects = {}
    frame_count = 0
    grace_period_frames = 30
    people_entering = {}
    entering = set()
    people_exiting = {}
    exiting = set()
    going_up = {}
    second_floor = set()
    going_down = {}
    first_floor = set()
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

        # Skip frames according to given rate
        if frame_count % 3 != 0:
            continue
            
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
                    # Increment frequency counter for whole bounding box
                    global_imgArray[y1:y2, x1:x2] += 1

                    # Check if object is entering
                    enterTest = cv2.pointPolygonTest(np.array(area4,np.int32),((x1,y2)),False)
                    if enterTest >= 0:
                        people_entering[id]=(x1,y2)
                    if id in people_entering:
                        confirmIn = cv2.pointPolygonTest(np.array(area3,np.int32),((x1,y2)),False)
                        if confirmIn >= 0:
                            entering.add(id)
                    
                    # Check if object is exiting
                    leaveTest = cv2.pointPolygonTest(np.array(area3,np.int32),((x1,y2)),False)
                    if leaveTest >= 0:
                        people_exiting[id]=(x1,y2)
                    if id in people_exiting:
                        confirmOut = cv2.pointPolygonTest(np.array(area4,np.int32),((x1,y2)),False)
                        if confirmOut >= 0:
                            exiting.add(id)

                    # Check if object is going up second floor
                    upTest = cv2.pointPolygonTest(np.array(area2,np.int32),((x1,y2)),False)
                    if upTest >= 0:
                        going_up[id]=(x1,y2)
                    if id in going_up:
                        confirmUp = cv2.pointPolygonTest(np.array(area1,np.int32),((x1,y2)),False)
                        if confirmUp >= 0:
                            second_floor.add(id)
                    
                    # Check if object is going down first floor
                    downTest = cv2.pointPolygonTest(np.array(area1,np.int32),((x1,y2)),False)
                    if downTest >= 0:
                        going_down[id]=(x1,y2)
                    if id in going_down:
                        confirmDown = cv2.pointPolygonTest(np.array(area2,np.int32),((x1,y2)),False)
                        if confirmDown >= 0:
                            first_floor.add(id)

                    # if we haven't seen a particular object ID before in entering, register it in a list 
                    if id in entering and id not in totalCount:
                        totalCount.append(id)

                    # Object is being tracked for the first time
                    if id not in tracked_objects:
                        tracked_objects[id] = {
                            'Track ID': id,
                            'Entry Time': frame_count / fps,
                            'Exit Time': None,
                            'Movement Tracks': [[centre_x, centre_y]],
                            'last_seen': frame_count,
                            'grace_period': 0,
                        }
                    # Object is still being tracked
                    else:
                        # Existing id, update movement tracks
                        movement_tracks = tracked_objects[id]['Movement Tracks']
                        movement_tracks.append([centre_x, centre_y])
                        tracked_objects[id]['last_seen'] += 1 # Get latest frame number
                        tracked_objects[id]['grace_period'] = 0 # Reset grace period frames

        # Handle objects that are not detected in the current frame
        for Id in list(tracked_objects.keys()):
            if Id not in [box.id for box in boxes]:
                # Id is lost and it is uncertain if it has exited
                if tracked_objects[Id]['Exit Time'] is None:
                    tracked_objects[Id]['grace_period'] += 1
                    # If grace period has expired, update the exit frame and time
                    if tracked_objects[Id]['grace_period'] >= grace_period_frames:
                        frame_exit = tracked_objects[Id]['last_seen']
                        tracked_objects[Id]['Exit Time'] = frame_exit / fps

        # Draw the area of interest (画出感兴趣的区域)
        # cv2.polylines(img,[np.array(area1,np.int32)],True,(255,0,0),2)
        # cv2.putText(img,str('1'),(28,265),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0,0,0),1)
        # cv2.polylines(img,[np.array(area2,np.int32)],True,(255,0,0),2)
        # cv2.putText(img,str('2'),(178,329),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0,0,0),1)
        cv2.polylines(img,[np.array(area3,np.int32)],True,(255,0,0),2)
        cv2.putText(img,str('3'),(587,465),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0,0,0),1)
        cv2.polylines(img,[np.array(area4,np.int32)],True,(255,0,0),2)
        cv2.putText(img,str('4'),(627,430),cv2.FONT_HERSHEY_SIMPLEX,(0.5),(0,0,0),1)


        # count_secFloor = (len(second_floor))
        # cv2.putText(img,"On 2nd floor: " + str(count_secFloor),(520,30),cv2.FONT_HERSHEY_SIMPLEX,(1),(0,0,0),2)
        count_txt = len(totalCount)
        count_leave = len(exiting)
        total = count_txt - count_leave
        cv2.putText(img,"Entering: " + str(count_txt),(520,30),cv2.FONT_HERSHEY_SIMPLEX,(1),(0,0,0),2)
        cv2.putText(img,"Leaving: " + str(count_leave),(520,80),cv2.FONT_HERSHEY_SIMPLEX,(1),(0,0,0),2)

        _record_movement_data(movement_data_writer,tracked_objects,movement_data_file)
        _record_crowd_data(crowd_data_writer,record_time,total)

        # Perform heatmap processing (执行热图处理)
        super_imposed_img = heatmap(img, global_imgArray)

        # Write the processed image to the video (将处理后的图像写入视频)   
        # output_video.write(img)

        # Show the processed image (显示处理后的图像)
        cv2.imshow("Processed Output", img)

        # Press q to exit (按q键退出)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # Close the video file (关闭视频文件)
    cv2.destroyAllWindows() # Close the window (关闭窗口)
    output_video.release() # Close video writing (关闭视频写入)