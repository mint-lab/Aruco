import json
import yaml
import numpy as np  
import glob
import cv2
from collections import OrderedDict
from pprint import pprint
CAMCHAIN_FILE ="metadata/2023-05-03-23-11-22-camchain.yaml"
IMAGE_FILE = 'output/images/test/*.png'
POSE_FILE = "metadata/pos.json"

def camchain2nerf(filepath):
    with open(filepath, 'r') as f:
        data = {}
        data = yaml.load(f,Loader= yaml.FullLoader)
        intrin = data['cam0']['intrinsics']
        
        dist = np.array((data['cam0']['distortion_coeffs']))

    return intrin,dist

       

def images2nerf(image_file_paths):
    image_paths = sorted(glob.glob(image_file_paths))
    cv_img = cv2.imread(image_paths[0])
    h, w, _ = cv_img.shape

    return image_paths,h,w
    

def pos2nerf(json_file): 
    with open(json_file,'r') as f:
            Plist = []
            json_pos= json.load(f)

            poses = json_pos["poses"]
            orientations = json_pos["orientations"]
            

            for ori, pos in zip(orientations, poses):
                ori = np.vstack((ori,[0,0,0]))
                pos = np.vstack((pos,[1]))
                P = np.hstack((ori,pos))
                Plist.append(P)
    return Plist

               



def intergrate():
    data_format = OrderedDict()
    intrin, dist = camchain2nerf(CAMCHAIN_FILE)
    k1 = dist[0]
    k2 = dist[1]
    p1 = dist[2]
    p2 = dist[3]

    fl_x = intrin[0]
    fl_y = intrin[1]
    cx = intrin[2]
    cy = intrin[3]

    file_paths ,h,w = images2nerf(IMAGE_FILE)
    camera_angle_x = 2 * np.tanh(2*fl_x / w)
    camera_angle_y = 2 * np.tanh(2*fl_y / h)

    data_format["camera_angle_x"] = camera_angle_x
    data_format["camera_angle_y"] = camera_angle_y
    data_format["fl_x"] = fl_x
    data_format["fl_y"] = fl_y
    data_format["k1"] = k1
    data_format["k2"] = k2
    data_format["p1"] = p1
    data_format["p2"] = p2
    data_format["cx"] = cx
    data_format["cy"] = cy
    data_format["w"] = w
    data_format["h"] =h 
    
    # frames 
    Plist = pos2nerf(POSE_FILE)
    frame_list = []
    for file_path,p in zip(file_paths, Plist):
        frame = dict()
        frame["file_path"] =file_path
        #frame["rotation"] = "rotation" 
        frame["transform_matrix"] = p.tolist()
        frame_list.append(frame)
    data_format["frames"] = frame_list
    
    # data_format = dict(data_format)

    return data_format


        
def dict2json(data):
    with open("metadata/data_format_half.json","w") as f:
        json.dump(data,f)

    

    

    
if __name__ =="__main__":

    ret = intergrate()
    dict2json(ret)
    

