import cv2
import numpy as np

# This file is used to get the coordinates on the image
def captureEvent(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x},{y})")

# area1 = [(284,365),(385,300),(416,305),(315,374)] # 1
# area2 = [(330,383),(428,312),(458,317),(367,394)] # 2
area1 = [(3,125),(27,124),(45,248),(17,248)] # 1
area2 = [(160,280),(430,240),(450,266),(180,310)] # 2
area3 = [(604,467),(730,540),(680,540),(576,478)] # 3
area4 = [(635,455),(662,443),(862,540),(790,540)] # 4

if __name__ == '__main__':
    img = cv2.imread("blockB.jpg")
    cv2.polylines(img,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.polylines(img,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.polylines(img,[np.array(area3,np.int32)],True,(255,0,0),2)
    cv2.polylines(img,[np.array(area4,np.int32)],True,(255,0,0),2)


    cv2.imshow("image", img)
    cv2.setMouseCallback("image", captureEvent)
    cv2.waitKey(0)
    cv2.destroyAllWindows()