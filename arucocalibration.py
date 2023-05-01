import cv2 
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json 
ARUCO_DICT =  {

    "DICT_4X4_50" : cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50" : cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100" : cv2.aruco.DICT_5X5_100
    
}


root =Path(__file__).parent.absolute()


arucoDict = aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_50"])

marker_length = 0.375
marker_gap = 0.5 

board = aruco.GridBoard((6,6),marker_length,marker_gap,arucoDict)

detectorParams = aruco.DetectorParameters()


camera = cv2.VideoCapture(0)
ret, img = camera.read()

ret, img = camera.read()
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
h,  w = img_gray.shape[:2]

pose_r, pose_t = [], []
while True:
    ret, img = camera.read()
    img_aruco = img
    im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = im_gray.shape[:2]
    dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
    #cv2.imshow("original", img_gray)
    if corners == None:
        print ("pass")
    else:

        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, newcameramtx, dist) # For a board
        print ("Rotation ", rvec, "Translation", tvec)
        if ret != 0:
            img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
            img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec, tvec, 10)    # axis length 100 can be changed according to your requirement

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break;
    cv2.imshow("World co-ordinate frame axes", img_aruco)

cv2.destroyAllWindows()