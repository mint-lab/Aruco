import cv2
import cv2.aruco as aruco 
import numpy as np 
import scipy 
from arucodetect import aruco_display
from numpy.linalg import norm 

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

marker_length = 0.099 # unit : m ... 50mm

def camera_pose_estimation(corners,ids, matrix_coefficients, distortion_coefficients, origin):

    """
    Estimate the camera pos in the World coordinates which the Origin is 
    Right top corner of ID:0

    This methods follows those steps.

    1) Find the Origin corner 
        *constraints* 
        aruco.detectMarkers returns corners and ids.
        Those are unordered and not able to make sure that detect the origin points. So some tricks are needed.

        1-1) 
        To run the code in a robust way, get a first-glanced corner 
        use the fact that each tag sqaures has same gap.

    2) Using aruco.estimatePoseSingleMarkers, get rvec and tvecs (marker to camera)

    3) returned rvec and tvec is the camera pose in the 3D world coordinates
    
    """
    if ids is None:
        return None
    
    if ids is not None:
        # Find the first captured id number
        first_captured_id = int(ids[0])
        real_length = 0.025 
        
        # Get camera pose from first captured tag  
        rvec, tvec, pts= aruco.estimatePoseSingleMarkers(corners[0], marker_length, matrix_coefficients, distortion_coefficients)
        tvec = tvec[0].reshape((3,1))
        R, jac = cv2.Rodrigues(rvec)
        cam_pos = -R.T@tvec
        cam_ori = R.T
        # Move camera_position  
        dx ,dy = int(first_captured_id/10), first_captured_id%10
        move = np.array([[dx*real_length],
                            [dy*real_length],
                            [0]])
        
        origin_cam_pos = cam_pos + move 

        return origin_cam_pos
    
    
    
    

    
        

    

aruco_dict = ARUCO_DICT["DICT_6X6_100"]
arucoDict = aruco.getPredefinedDictionary(aruco_dict)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(arucoDict,detectorParams)

intrinsic_camera = np.array(((558,0,1280),(0,539,720),(0,0,1)))
dist = np.array((
-0.04347546948864125,
0.00041693359722417287,
 -0.001068504464681613,
 0.013606169764053481,
 -6.822152395057791e-07))


# Camera pose Estimation witqh ArUco tags.
rvecs = []
tvecs = []

# Cofig Camera
usb_camera = 2
cap =cv2.VideoCapture(usb_camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# init ArUco
arucoDict = aruco.getPredefinedDictionary(aruco_dict)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(arucoDict,detectorParams)

# While camera is opened 
while cap.isOpened():
    # Capture th videdo
    _, img = cap.read()

    # Convert images to gray 
    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers 
    corners ,ids, rejected = detector.detectMarkers(gray)    

    # visualize detected tags and origin
    origin_pixel = aruco_display(corners, ids, rejected, img)

    # recoginize Origin points and return camera pose and origin 
    cam_pos  = camera_pose_estimation(corners,ids, intrinsic_camera, dist, origin_pixel)

    # visualize origin 
    if cam_pos is not None:
        pose_string = f'XYZ [{cam_pos[0][0]:.3f},{cam_pos[1][0]:.3f},{cam_pos[2][0]:.3f}]'
        cv2.putText(img,pose_string,(5,16),cv2.FONT_ITALIC,1,(255,100,0),cv2.LINE_8)
    elif cam_pos is None: 
        cv2.putText(img,"Origin cannot be detected",(5,16),cv2.FONT_ITALIC,1,(255,100,0),cv2.LINE_8)
        
    cv2.imshow("result" ,img) 

    key =cv2.waitKey(1) & 0xFF 
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

    
