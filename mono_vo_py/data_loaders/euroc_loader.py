import os
import glob
import yaml
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation


# TODO: sync image and imu data
class EuRoCLoader:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root


    def load_image_file_name(self):
        # Load image filenames (sorted)
        self.cam_folder = os.path.join(self.dataset_root, 'cam0', 'data')
        self.image_files = sorted(glob.glob(os.path.join(self.cam_folder, '*.png')))
        return self.image_files


    def load_intrinsics(self):
        # intrinsic parameters:        
        sensor_config_path = os.path.join(self.dataset_root, "cam0", "sensor.yaml")
        with open(sensor_config_path) as f:
            sensor_config = yaml.safe_load(f)
        return sensor_config["intrinsics"]

    def load_gt_pose(self):
        gt_pose_dir = os.path.join(self.dataset_root, 
                                    "state_groundtruth_estimate0", 
                                    "data.csv")
        # read ground truth
        with open(gt_pose_dir) as f:
            gt_pose_str_list = f.readlines()
        
        gt_pose_list = []
        for i in range(1, len(gt_pose_str_list)):
            T_wc_vec = gt_pose_str_list[i].strip().split(",")[:8]
            T_wc_vec = np.array([float(x) for x in T_wc_vec])
            p_wc = T_wc_vec[1:4]
            q_wc = T_wc_vec[4:8]
            T_wc = np.eye(4)
            T_wc[:3, :3] = Rotation.from_quat(q_wc).as_matrix()
            T_wc[:3, 3] = p_wc
            gt_pose_list.append(T_wc)
        
        return gt_pose_list


    def load_pcd(self):
        pcd_file = os.path.join(self.dataset_root, "pointcloud0", "data.ply")
        pcd = o3d.io.read_point_cloud(pcd_file)
        return pcd


    def load_imu_data(self):
        # Load IMU data as numpy array
        imu_file = os.path.join(self.dataset_root, 'imu0', 'data.csv')
        imu_df = pd.read_csv(imu_file, comment='#', header=None)
        imu_df.columns = [
            'timestamp',
            'w_x', 'w_y', 'w_z',
            'a_x', 'a_y', 'a_z'
        ]
        self.imu_data = imu_df.to_numpy()
        return self.imu_data