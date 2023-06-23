from ultralytics import YOLO 
import math
import cv2 
import cvzone
import numpy as np
import os
import imutils
import deep_sort.deep_sort.deep_sort as ds


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



def processVideo(inputPath,model):
    #Load deepsort weight file (加载deepsort权重文件)
    tracker = ds.DeepSort('deep_sort/deep_sort/deep/checkpoint/ckpt.t7')
    
    #Load YOLO model file (加载YOLO模型文件)
    model=YOLO(model)
    #Read the video from inputPath (从inputPath读入视频)
    cap = cv2.VideoCapture(inputPath)

    #Get the frame rate of the video (获取视频的帧率)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    #Get the size of the video (获取视频的大小)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    #Initialize video writing
    output_video = cv2.VideoWriter()
    outputPath = 'output_videos'
    os.makedirs(outputPath, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    #Create output video path (创建输出视频路径)
    video_save_path = os.path.join(outputPath,"output.mp4")
    
    output_video.open(video_save_path , codec, fps, size, True)

    #Read and process each frame of the picture(对每一帧图片进行读取和处理)
    while True:
        #Read frame from video
        ret, img = cap.read()
        #Resize Frame to half of its size for faster detection
        img = imutils.resize(img, width=(int(img.shape[1] / 2)), height=(int(img.shape[0] / 2)))
        #Infer each frame of the picture
        results=model(img)
        #Initialize detection box by size 0 * 4
        detections=np.empty((0, 4))
        #Initialize confidence array
        confarray = []

        #Exit if no picture is read (如果没有读取到图片则退出)
        if not(ret):
            break

        #Read the inferred data (读取推理的数据)
        for r in results:
            boxes=r.boxes
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
        resultsTracker=tracker.update(detections, confarray, img)
        for result in resultsTracker:
            x1,y1,x2,y2,Id=result
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            #Draw tracking box (画跟踪框)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),1)
            #Write ID on the tracking box (在跟踪框上写入ID)
            cvzone.putTextRect(img,f'{int(Id)}',(max(-10,x1),max(40,y1)),scale=0.5,thickness=1, 
                               offset=1, font=cv2.FONT_HERSHEY_DUPLEX)

        # Write the processed image to the video (将处理后的图像写入视频)   
        # output_video.write(img)

        # Show the processed image (显示处理后的图像)
        cv2.imshow("Processed Output",img)

        # Press q to exit (按q键退出)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # output_video.release() # Close video writing (关闭视频写入)
    cap.release() # Close the video file (关闭视频文件)
    cv2.destroyAllWindows() # Close the window (关闭窗口)


if __name__ == '__main__':
    model = "yolov8l.pt"
    input_video_path = "BusStop2.MOV"
    processVideo(input_video_path, model)