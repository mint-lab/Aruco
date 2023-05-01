import cv2
import cv2.aruco as aruco 

ARUCO_DICT =  {
    "DICT_4X4_50" : cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50" : cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100" : cv2.aruco.DICT_5X5_100
    
}

def aruco_display (corners, ids ,rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

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
            cv2.circle(image,(cX,cY),4,(0,0,255),-1)

    return image

if __name__ == '__main__':

    arucoDict = aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_50"])
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict,detectorParams)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()

        h,w,_ = img.shape
        width = 1000
        height = int(width*(h/w))
        img = cv2.resize(img,(width,height), interpolation=cv2.INTER_CUBIC)
        corners ,ids, rejected = detector.detectMarkers(img)

        detected_markers = aruco_display(corners,ids,rejected,img)
        
        cv2.imshow("Result",detected_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()