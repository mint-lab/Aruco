# Aruco



## 1 arucopose.py / .cpp
- Using cv2.EstimatePoseboard, camera pose will be estimated.
- The thing is that the Method (cv2.Est...) has apposite origin direction vectors that we want to
- This can results in wrong rendering like the sampled ray colors flipped 
- So Makeup Matrix will be required to flip the estimated Origin
- output -> pos.json

## 2 pos_to_nerf.py / .cpp 

- This code intergrate 
  - pos.json [result of pose estimation]
  - cam_chain [calibration results]
  - image paths

- The output form of this code fit in the metadata format that colmap/instant-ngp required 