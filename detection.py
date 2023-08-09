from ultralytics import YOLO 
import math
import cv2 
import cvzone
import numpy as np
import os
import imutils
from colors import RGB_COLORS
import deep_sort.deep_sort.deep_sort as ds
import datetime
import pandas as pd

#YoloV8 official model label data, this project only uses 'person'
#YoloV8官方模型标签数据，本次项目只使用了'person'
# classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite',
#     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut',
#     'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#     'scissors',
#     'teddy bear', 'hair drier', 'toothbrush']

IF_CAM = False

# Save dict of tracked objects into a csv file (将跟踪对象的字典保存到csv文件中)
def _record_movement_data(movement_data_writer,tracked_objects,file):
    for track_id in tracked_objects.keys():
        entry_time = tracked_objects[track_id]['Entry Time']
        exit_time = tracked_objects[track_id]['Exit Time']
        movement_tracks = tracked_objects[track_id]['Movement Tracks']
        movement_data_writer.writerow([track_id, entry_time, exit_time, movement_tracks])
        
    file.flush()
    # os.fsync(file.fileno())

    movement_data_writer.writerow([])
    file.flush()
    # os.fsync(file.fileno())
    
    df = pd.read_csv('processed_data/movement_data.csv')
    unique_df = df.dropna(subset=['Track ID'])
    # Drop duplicate rows based on the 'ID' column, keeping only the last occurrence (most recent row)
    unique_df = df.drop_duplicates(subset=['Track ID'], keep='last')
    unique_df.to_csv('processed_data/movement_data.csv', index=False)


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


def processVideo(inputPath,model,movement_data_writer,movement_data_file):
    #Load deepsort weight file (加载deepsort权重文件)
    tracker = ds.DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7')
    
    #Load YOLO model file (加载YOLO模型文件)
    model=YOLO(model)
    model.to('cuda')
    #Read the video from inputPath (从inputPath读入视频)
    cap = cv2.VideoCapture(inputPath)

    #Get the frame rate of the video (获取视频的帧率)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    #Get the size of the video (获取视频的大小)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h)
    
    #Initialize video writing
    output_video = cv2.VideoWriter()
    outputPath = 'output_videos'
    os.makedirs(outputPath, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    #Create output video path (创建输出视频路径)
    video_save_path = os.path.join(outputPath,"output.mp4")
    output_video.open(video_save_path , codec, fps, size, True)

    global_imgArray = None
    global_imgArray = np.ones([int(h), int(w)], dtype=np.uint32)

    # Initialize tracked_objects dictionary
    tracked_objects = {}
    # Number of frames to wait before marking an object as truly exited
    grace_period_frames = 60
    frame_count = 0

    #Read and process each frame of the picture(对每一帧图片进行读取和处理)
    while cap.isOpened():
        #Read frame from video
        ret, img = cap.read()

        #Exit if no picture is read (如果没有读取到图片则退出)
        if not ret:
            break

        #Infer each frame of the picture
        results = model(img, verbose=False)
        #Initialize detection box by size 0 * 4
        detections = np.empty((0, 4))
        #Initialize confidence array
        confarray = []
        
        if IF_CAM:
            current_datetime = datetime.datetime.now()
        else:
            # Get the current video time (获取当前视频时间)
            current_datetime = frame_count

        #Read the inferred data (读取推理的数据)
        for r in results:
            boxes = r.boxes
            # For each detection box in the frame (对帧中的每个检测框)
            for box in boxes:
                #Get the coordinates of the upper left and lower right corner of the box
                #(获取框的左上角和右下角坐标)
                x1,y1,x2,y2=box.xywh[0]
                #Convert tensor type to integer (将tensor类型转变为整型)
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                #Take 2 decimal places for conf (对conf取2位小数)
                conf=math.ceil(box.conf[0]*100)/100
                #Get the object category label (获取物体类别标签)
                cls=int(box.cls[0])
                
                #Only detect and track person class (只检测和跟踪行人)
                if cls==0:
                    currentArray=np.array([x1,y1,x2,y2])
                    confarray.append(conf)
                    #Stack data by row (按行堆叠数据)
                    detections=np.vstack((detections,currentArray))

        #行人跟踪
        resultsTracker = tracker.update(detections, confarray, img)
        for result in resultsTracker:
            x1,y1,x2,y2,Id=result
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            #Draw tracking box (画跟踪框)
            cv2.rectangle(img,(x1,y1),(x2,y2), RGB_COLORS["blue"], 1)
            #Write ID on the tracking box (在跟踪框上写入ID)
            cvzone.putTextRect(img,f'{int(Id)}',(max(-10,x1),max(40,y1)),scale=0.8,thickness=1, 
                               colorR=RGB_COLORS["blue"], offset=1, font=cv2.FONT_HERSHEY_DUPLEX)
            # Increment frequency counter for whole bounding box (增加整个边界框的频率计数器)
            global_imgArray[y1:y2, x1:x2] += 1

            # Update tracked_objects dictionary with centroid and exit frame
            centre_x = int((x1 + x2) / 2)
            centre_y = int((y1 + y2) / 2)
            # Object is being tracked for the first time (物体第一次被追踪)
            if Id not in tracked_objects:
                tracked_objects[Id] = {
                    'Track ID': Id,
                    'Entry Time': frame_count / fps,
                    'Exit Time': None,
                    'Movement Tracks': [[centre_x, centre_y]],
                    'grace_period': 0,
                }
            # Object is still being tracked (物体仍在被追踪)
            else:
                # Existing id, update movement tracks
                movement_tracks = tracked_objects[Id]['Movement Tracks']
                movement_tracks.append([centre_x, centre_y])
                tracked_objects[Id]['grace_period'] = 0 # Reset grace period frames

        # Handle objects that are not detected in the current frame 
        for Id in list(tracked_objects.keys()):
            if Id not in [result[4] for result in resultsTracker]:
                # Id is lost and it is uncertain if it has exited
                if tracked_objects[Id]['Exit Time'] is None:
                    tracked_objects[Id]['grace_period'] += 1
                # If grace period has expired, update the exit frame and time
                if tracked_objects[Id]['grace_period'] >= grace_period_frames:
                    tracked_objects[Id]['Exit Time'] = frame_count / fps

        # Increment frame count
        frame_count += 1
        
        _record_movement_data(movement_data_writer,tracked_objects,movement_data_file)

        # Perform heatmap processing (执行热图处理)
        super_imposed_img = heatmap(img, global_imgArray)
                        
        #Resize Frame to half of its size for faster display in imshow (将帧的大小调整为其一半，以便在imshow中更快地显示)
        resize = imutils.resize(super_imposed_img, width=(int(super_imposed_img.shape[1] / 2)), 
                             height=(int(super_imposed_img.shape[0] / 2)))

        # Write the processed image to the video (将处理后的图像写入视频)   
        #output_video.write(super_imposed_img)

        # Show the processed image (显示处理后的图像)
        cv2.imshow("Processed Output",resize)

        # Press q to exit (按q键退出)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() # Close the video file (关闭视频文件)
    cv2.destroyAllWindows() # Close the window (关闭窗口)
    output_video.release() # Close video writing (关闭视频写入)



# if __name__ == '__main__':
#     model = "yolov8x.pt"
#     input_video_path = "Library.MOV"
#     processVideo(input_video_path, model)