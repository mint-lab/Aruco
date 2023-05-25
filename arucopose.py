import cv2
import cv2.aruco as aruco 
import numpy as np 
import scipy 
from visualization.arucodetect import aruco_display
import json
import yaml
from pprint import pprint
from collections import defaultdict
from scipy.spatial.transform import Rotation as R 
import argparse
import glob
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

marker_length = 0.052# unit : m ...
   

def camera_pose_board(corners, ids, K, dist, arucoDict,img):
    
    gap = 0.005
    markerLength = 0.052
    rvec = None
    tvec = None
    board = aruco.GridBoard((5,7),markerLength, gap, arucoDict)
    
    _, rvec, tvec= aruco.estimatePoseBoard(corners, ids, board, K, dist, rvec, tvec)
    
    Rmat, _ = cv2.Rodrigues(rvec.flatten())
    M = R.from_rotvec(np.pi*np.array([-1,0,0]))
    M = M.as_matrix()
	# estimatePoseBoard axis was flipped so modification will be required 
	# Make up rotation matrix

    cam_pos = -Rmat.T@tvec
    cam_ori = Rmat.T  
    cam_ori = cam_ori @ M
    
    
    return cam_pos , cam_ori
    
if __name__ =="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--type","-t",type=str,default="def")
    # args = parser.parse_args()
    
    # Init AruCo
    aruco_dict = ARUCO_DICT["DICT_6X6_250"] 

    with open("metadata/2023-05-03-23-11-22-camchain.yaml", 'r') as f:
        data = {}
        data = yaml.load(f,Loader= yaml.FullLoader)
        intrin = data['cam0']['intrinsics']
        
        dist = np.array((data['cam0']['distortion_coeffs']))
        K = np.array(((intrin[0], 0, intrin[2]), 
                    (0, intrin[1], intrin[3]), 
                    (0,         0,         1)), dtype = np.float32)                    
        
    # Camera pose Estimation witqh ArUco tags.
    ret = defaultdict(list)
    # init ArUco
    arucoDict = aruco.getPredefinedDictionary(aruco_dict)
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict,detectorParams)

    images = sorted(glob.glob('output/images/test/*.png'))
    print("estimation proceedings..")
    for image in images:
        
        img = cv2.imread(image)
        
        h, w,_ = img.shape
        # Convert images to gray 
        gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers 
        corners ,ids, rejected = detector.detectMarkers(gray)   

        # visualize detected tags and origin
        origin_pixel = aruco_display(corners, ids, rejected, img)
        
        pos, ori = camera_pose_board(corners, ids, K, dist, arucoDict, img)
        
        cv2.imshow("res",img)    
        
        ret["poses"].append(pos.tolist())
        ret["orientations"].append(ori.tolist())

        key =cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            cv2.destroyAllWindows()


    print("- finished - ")
    with open('metadata/pos.json', 'w') as f:
        json.dump(ret,fp=f)
        
