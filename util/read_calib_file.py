from pathlib import Path
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
import numpy as np
import yaml

def read_calib_file(filepath: str)->dict:
    """Read in a calibration file and parse into a dictionary."""
    data = {}
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    return data

def parsing_calib_data_kalibr2cv(calib_dict: dict, base_cam: str = 'cam0') -> dict:
    """Parse the calibration data from Kalibr to OpenCV format."""
    calib_dict_cv = {}
    # intrinsics
    calib_dict_cv['intrinsics'] = np.array(calib_dict[base_cam]['intrinsics'])
    calib_dict_cv['resolution'] = calib_dict[base_cam]['resolution']
    calib_dict_cv['focal_x'] = calib_dict_cv['intrinsics'][0]
    calib_dict_cv['focal_y'] = calib_dict_cv['intrinsics'][1]
    calib_dict_cv['center_x'] = calib_dict_cv['intrinsics'][2]
    calib_dict_cv['center_y'] = calib_dict_cv['intrinsics'][3]
    calib_dict_cv['width'] = calib_dict_cv['resolution'][0]
    calib_dict_cv['height'] = calib_dict_cv['resolution'][1]
    calib_dict_cv['K'] = np.array([[calib_dict_cv['focal_x'], 0, calib_dict_cv['center_x']],
                                   [0, calib_dict_cv['focal_y'], calib_dict_cv['center_y']],
                                   [0, 0, 1]], dtype=np.float32)
    calib_dict_cv['dists'] = np.array(calib_dict[base_cam]['distortion_coeffs'])

    # extrinsics
    if base_cam != 'cam0':
        calib_dict_cv['T2base'] = np.array(calib_dict[base_cam]['T_cn_cnm1'])
        calib_dict_cv['R2base'] = R.from_matrix(calib_dict_cv['T2base'][:3, :3]).as_rotvec()
        calib_dict_cv['t2base'] = calib_dict_cv['T2base'][:3, 3]

    return calib_dict_cv


def load_calibed_data(calib_yaml_path: str, base_cam: str = 'cam0') -> dict:
    """Load the calibration data from yaml file and parse to OpenCV format."""
    calib_yaml = read_calib_file(calib_yaml_path)
    calib_dict = parsing_calib_data_kalibr2cv(calib_yaml, base_cam)

    return calib_dict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--calib_file', '-cf', type=str, default='calib.yaml', help='path to calibration file')
    parser.add_argument('--base_cam', '-bc', type=str, default='cam0', help='base camera')
    args = parser.parse_args()

    calib_yaml_path = Path().cwd() / args.calib_file
    base_cam = args.base_cam

    calib_yaml = load_calibed_data(calib_yaml_path, base_cam)

    print(calib_yaml)