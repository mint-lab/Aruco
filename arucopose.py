import cv2
import cv2.aruco as aruco 
import numpy as np 
from arucodetect import aruco_display
from scipy.spatial.transform import Rotation
from collections import defaultdict
ARUCO_DICT =  {

    "DICT_4X4_50" : cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50" : cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100" : cv2.aruco.DICT_5X5_100
    
}

marker_length = 0.05 # unit : m ... 50mm

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    
    global rvec
    global tvec
    gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.getPredefinedDictionary(aruco_dict_type)
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict,detectorParams)

    corners ,ids, rejected = detector.detectMarkers(gray)
    rvecs = []
    tvecs = []
    if len(corners):
        for i in range(0,len(ids)):
          
            rvec,tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_length, matrix_coefficients,
                                                                        distortion_coefficients)
            
            #aruco.drawdetectedMarkers(frame,corners)

            detected_markers = aruco_display(corners,ids,rejected,img)
            #aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec)
            rvecs.append(rvec)
            tvecs.append(tvec)
    return frame, rvecs, tvecs


arucoDict = aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_50"])
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(arucoDict,detectorParams)

intrinsic_camera = np.array(((558,0,1280),(0,539,720),(0,0,1)))
dist = np.array((
-0.04347546948864125,
0.00041693359722417287,
 -0.001068504464681613,
 0.013606169764053481,
 -6.822152395057791e-07))


# Camera pose Estimation with ArUco tags.
rvecs = []
tvecs = []
cap =cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

while cap.isOpened():
    ret, img = cap.read()
    
    output, rvec, tvec = pose_estimation(img, ARUCO_DICT["DICT_5X5_50"], intrinsic_camera, dist)

    cv2.imshow("Estimated Pose", output)
    
    print(tvec)
    key =cv2.waitKey(1) & 0xFF 
    if key == 32: # Space Key 
        rvecs.append(rvec)
        tvecs.append(tvec)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# Save to text file 

with open('result.txt', 'w') as f:
    for i,(pos) in enumerate(zip(rvecs,tvecs)):
        r, t = pos

        r ,t = r[0].flatten(), t[0].flatten()
        f.write(f"view {i+1}\n")
        f.write(f"rvec : {r}")
        f.write('\n')
        f.write(f"tvec : {t}")
        f.write('\n')

    
