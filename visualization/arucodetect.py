import cv2
import cv2.aruco as aruco 
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
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display (corners, ids ,rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4,2))
            (topRight, bottomRight, bottomLeft, topLeft) = corners

            topRight = (int(topRight[0]),int(topRight[1]))
            bottomRight = (int(bottomRight[0]),int(bottomRight[1]))
            topLeft = (int(topLeft[0]),int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]),int(bottomLeft[1]))

            cv2.line(image, topLeft, topRight, (0,255,0), 2)
            cv2.line(image, topRight, bottomRight, (0,255,0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0,255,0), 2)
            cv2.line(image, bottomLeft, topLeft, (0,255,0), 2)

            cX = int((topLeft[0] + bottomRight[0])/2.0)
            cY = int((topLeft[1] + bottomRight[1])/2.0)
    
            cv2.putText(image, str(markerID),(cX,cY),cv2.FONT_ITALIC,1,(0,0,255),cv2.LINE_4)
            cv2.circle(image,(cX,cY),4,(0,0,255),-1)

            if markerID == 0: # Draw Origin
                cv2.putText(image, "Origin",(topRight),cv2.FONT_ITALIC,1,(255,0,0),cv2.LINE_4)
                cv2.putText(image, f'{topRight[0], topRight[1]}',(topRight[0]+10,topRight[1]+10),cv2.FONT_ITALIC,1,(255,0,0),cv2.LINE_4)
                cv2.circle(image,(topRight),6,(255,0,0),-1)


if __name__ == '__main__':

    arucoDict = aruco.getPredefinedDictionary(ARUCO_DICT["DICT_6X6_50"])
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict,detectorParams)
    
    web_cam = 0
    usb_camera = 2 
    cap = cv2.VideoCapture(usb_camera)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    images = sorted(glob.glob('output/images/d435i_imgs/*.png'))
    print("estimation proceedings..")

    for image in images:    
        img = cv2.imread(image)
        h,w,_ = img.shape
        width = 1000
        height = int(width*(h/w))
        img = cv2.resize(img,(width,height), interpolation=cv2.INTER_CUBIC)
        corners ,ids, rejected = detector.detectMarkers(img)
    
        aruco_display(corners,ids,rejected,img)
        
        cv2.imshow("Result",img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()