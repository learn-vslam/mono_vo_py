import os
import glob
import yaml
import numpy as np


class KITTILoader:
    def __init__(self, config):
        self.config = config


    def load_img_files(self):
        img_data_folder = os.path.join(self.config.data_root,
                                    'kitti-odom/data_odometry_color/dataset/sequences/', 
                                    self.config.seq, 'image_2/')
        img_file_names = sorted(glob.glob(os.path.join(img_data_folder, "*.png")))
        return img_file_names
    

    def load_intrinsics(self):
        # intrinsic parameters:        
        config_path = './config/kitti_odom.yaml'
        with open(config_path) as f:
            cam_config = yaml.safe_load(f)
        return cam_config
    

    def load_gt_pose(self):
        gt_pose_dir = os.path.join(self.config.data_root, 
                                    'kitti-odom/data_odometry_poses/dataset/poses/', 
                                    self.config.seq + '.txt')
        # read ground truth
        with open(gt_pose_dir) as f:
            gt_pose_str_list = f.readlines()
        
        gt_pose_list = []
        for i in range(len(gt_pose_str_list)):
            T_wc_vec = gt_pose_str_list[i].strip().split()
            T_wc_vec = np.array([float(x) for x in T_wc_vec])
            T_wc = np.reshape(T_wc_vec, (3, 4))
            T_wc = np.concatenate((T_wc, np.array([[0, 0, 0, 1]])), axis=0)
            gt_pose_list.append(T_wc)

        return gt_pose_list