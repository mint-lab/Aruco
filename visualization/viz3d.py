import numpy as np
import yaml
import json 
from pprint import pprint
import cv2
import open3d as o3d

def show_result(camera_list):
    o3d.visualization.draw_geometries(camera_list)

def pyramid():
    points = [
    [-.5, .5, 0],
    [.5,  .5, 0],
    [.5, -.5, 0],
    [-.5,-.5, 0],
    [0,    0, 1]
                ]
    lines = [
        [0, 1],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4]
             ]
    
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

if __name__ == "__main__":

        
    
    with open('data/transforms.json','r') as f:
            json_pos= json.load(f)
            
            frames = json_pos["frames"]

            poses = []
            orientations = []
            for fr in frames:
                tr = fr["transform_matrix"]
                tr_np = np.array(tr)
                ps = tr_np[:3,3:].ravel().tolist()
                ori = tr_np[:3,:3].tolist()
                
                poses.append(ps)
                orientations.append(ori)

    with open("metadata/2023-05-03-23-11-22-camchain.yaml", 'r') as f:
        data = {}
        data = yaml.load(f,Loader= yaml.FullLoader)
        intrin = data['cam0']['intrinsics']
        
        dist = np.array((data['cam0']['distortion_coeffs']))
        K = np.array(((intrin[0], 0, intrin[2]), 
                    (0, intrin[1], intrin[3]), 
                    (0,         0,         1)), dtype = np.float32)        

    poses = np.asarray(poses)
    orientations = np.asarray(orientations)
    NUM_OF_VIEWS = len(poses)
    
    # Preprocessing for pointCloud : should be (N x 3) shape and get xyz euler angles from R matrix 
    poses = poses.reshape((-1,3))
    # Init coordinates 
    #camera_list = [o3d.geometry.TriangleMesh.create_coordinate_frame() for i in range(NUM_OF_VIEWS)]
    
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
    origin = origin.scale(0.05,center=origin.get_center())

    camera_list = [origin]

    for i in range(NUM_OF_VIEWS):
        camera_list.append(pyramid())

    
    
    for i in range(1,NUM_OF_VIEWS+1):
        camera_list[i] = camera_list[i].translate((poses[i-1][0],poses[i-1][1],poses[i-1][2]))
        camera_list[i] = camera_list[i].scale(0.02,center=camera_list[i].get_center())
        camera_list[i] = camera_list[i].rotate(orientations[i-1])
    
    # Show Result 
    
    show_result(camera_list)
    

    
    
        