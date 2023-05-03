from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
import numpy as np
import yaml, pprint

Kalibr_keywords = ["intrinsics", "resolution", "distortion_coeffs", "T_cn_cnm1"]
Kalibr_cam_names = ["cam0", "cam1", "cam2", "cam3", "cam4"]
MINT_keywords = {
    "cam0": "d435i",
    "cam1": "t265_left",
    "cam2": "t265_right",
    "cam3": "zedm_left",
    "cam4": "zedm_right",
    "camera_model": ["pinhole-radtan", "pinhole-equi"],
    "intrinsics": ["K", "fx", "fy", "cx", "cy"],
    "resolution": ["width", "height"],
    "distortion_coeffs": "dists",
    "T_cn_cnm1": "T_cm2cm0",
}

example_mint_format = {
    "d435i": {
        "cam_model": "pinhole-radtan",
        "K": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "dists": [1, 2, 3, 4, 5],
        "fx": 1,
        "fy": 2,
        "cx": 3,
        "cy": 4,
    },
    "t265_left": {
        "cam_model": "pinhole-radtan",
        "K": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "dists": [1, 2, 3, 4, 5],
        "fx": 1,
        "fy": 2,
        "cx": 3,
        "cy": 4,
        "T_cm2cm0": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
}

K_from_4params = lambda intr: np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], dtype=np.float32)

def kalibr2mint(filepath: str) -> dict:
    """Convert the kalibr calibration file to mint format."""
    data = read_calib_file(filepath)
    mint_data = {}
    for cam_name in Kalibr_cam_names:
        if cam_name in data:
            mint_data[MINT_keywords[cam_name]] = {}
            for keyword in MINT_keywords:
                if keyword == "camera_model":
                    mint_data[MINT_keywords[cam_name]][keyword] = data[cam_name][keyword] + "-" + data[cam_name]["distortion_model"]
                elif keyword == "intrinsics":
                    mint_data[MINT_keywords[cam_name]][keyword] = {}
                    for intrin in MINT_keywords[keyword]:
                        if intrin == "K":
                            mint_data[MINT_keywords[cam_name]][keyword][intrin] = K_from_4params(data[cam_name][keyword])
                        elif intrin == "fx":
                            mint_data[MINT_keywords[cam_name]][keyword][intrin] = data[cam_name][keyword][0]
                        elif intrin == "fy":
                            mint_data[MINT_keywords[cam_name]][keyword][intrin] = data[cam_name][keyword][1]
                        elif intrin == "cx":
                            mint_data[MINT_keywords[cam_name]][keyword][intrin] = data[cam_name][keyword][2]
                        elif intrin == "cy":
                            mint_data[MINT_keywords[cam_name]][keyword][intrin] = data[cam_name][keyword][3]
                    
                elif keyword == "resolution":
                    mint_data[MINT_keywords[cam_name]][keyword] = data[cam_name][keyword]
                
                elif keyword == "distortion_coeffs":
                    mint_data[MINT_keywords[cam_name]][keyword] = np.array(data[cam_name][keyword], dtype=np.float32)
                
                elif keyword == "T_cn_cnm1":
                    if cam_name != "cam0":
                        mint_data[MINT_keywords[cam_name]][keyword] = np.array(data[cam_name][keyword], dtype=np.float32)

    return mint_data

def read_kalibr_yaml(filepath: str)->dict:
    """Read in a calibration file in kalibr format and parse into a dictionary with the keywords."""
    data = {}
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    return data

def read_calib_file(filepath: str)->dict:
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--calib_file', '-cf', type=str, default='calib.yaml', help='path to calibration file')
    parser.add_argument('--base_cam', '-bc', type=str, default='cam0', help='base camera')
    args = parser.parse_args()

    calib_yaml_path = Path().cwd() / args.calib_file

    mint_calibed_data = kalibr2mint(calib_yaml_path)
    pprint.pprint(mint_calibed_data)

    # print("=====================================")
    # for tmp in MINT_keywords:
        # print(tmp)