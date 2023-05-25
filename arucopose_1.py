import pyrealsense2 as rs
import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation as R 

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)


arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.GridBoard((5, 7), 0.0515, 0.005, arucoDict)
detectorParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict,detectorParams)

pipeline.start(config)

intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = np.float32([[607.1324134094422, 0, 322.2877749770051],
                [0, 604.7435541260728, 252.4706756878432],
                [0, 0, 1]])
dist = np.float32([0.1222691517123698, -0.2731539846366536, 0.009788785415029547, -0.0010727804035390756])
rvec = None
tvec = None

images = []
matrixs = []

ret = False
n_images = 0

matrix = np.eye(4, 4, dtype=np.float32)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    image = color_image.copy()
    corners ,ids, rejected = detector.detectMarkers(image)     
    if ids is not None:
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        ret, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, K, dist, rvec, tvec)
        Rmat, _ = cv2.Rodrigues(rvec.flatten())
    else:
        ret = False
    cv2.imshow("color", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        if ret:
            images.append(color_image)
            M = R.from_rotvec(np.pi*np.array([-1, 0, 0]))
            M = M.as_matrix()
            
            Rmat = Rmat.T @ M
            tvec = -Rmat.T @ tvec
            matrix[:3, :3] = Rmat
            matrix[:3, 3:] = tvec
            matrixs.append(matrix.tolist())
            n_images += 1

cv2.destroyAllWindows()
pipeline.stop()

w = 640
h = 480

camera_angle_x = 2 * np.arctan(0.5 * w / K[0][0])
camera_angle_y = 2 * np.arctan(0.5 * w / K[1][1])

data = {
    "camera_angle_x" : camera_angle_x,
    "camera_angle_y" : camera_angle_y,
    "fl_x" : K[0][0].astype(np.float64),
    "fl_y" : K[1][1].astype(np.float64),
    "frames" : [
        {
            "file_path" : "images/%06d.png" % i,
            "transform_matrix" : matrixs[i]
        } for i in range(n_images)
    ]
}

for i in range(n_images):
    cv2.imwrite("data/images/%06d.png" % i, images[i])

with open("data/transforms.json", "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4)