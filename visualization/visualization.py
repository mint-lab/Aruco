"""
visaulization in order to make sure the camera pose estimation was worked

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import json
import yaml
from pprint import pprint 
import numpy as np
poses = []
orientations = []


with open('pos.json','r') as f:
        json_pos= json.load(f)

        poses = json_pos["poses"]
        orientations = json_pos["orientations"]


with open("2023-05-03-23-11-22-camchain.yaml", 'r') as f:
    data = {}
    data = yaml.load(f,Loader= yaml.FullLoader)
    intrin = data['cam0']['intrinsics']
    
    dist = np.array((data['cam0']['distortion_coeffs']))
    K = np.array(((intrin[0], 0, intrin[2]), 
                 (0, intrin[1], intrin[3]), 
                 (0,         0,         1)), dtype = np.float32)                    
    



def render_cam(): 
    fig = plt.figure()
    ax =fig.add_subplot(projection ='3d') 

    
    # Point Origin 
    ax.scatter(0,0,0, c="r")
    ax.legend("O")

    #Draw camera poses
    for ori,pos in zip(orientations,poses):
        ax.scatter(pos[0],pos[1],pos[2],c='b')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.show()
    plt.pause(300)
    plt.savefig("result_board.png",dpi =300)
    

    
    
         
        

if __name__ == "__main__":
     render_cam()
    

    
    


    

    


