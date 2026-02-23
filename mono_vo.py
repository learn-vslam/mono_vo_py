import os
import sys
import argparse
import time
from glob import glob
import numpy as np
import random
import cv2
from scipy.spatial.transform import Rotation
import tyro 
from dataclasses import dataclass
import viser
import viser.transforms as tf
import yaml
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_loaders.kitti_loader import KITTILoader
from data_loaders.euroc_loader import EuRoCLoader


@dataclass
class Config:
    data_root = os.path.join("/media", os.environ["USER"], "SeagateDrive/ws/datasets/")
    dataset_type: str = 'KITTI' # ['KITTI', "EuRoC"]
    load_gt_pose: bool = True
    seq: str = '09' # '00'
    key_frame_interval = 5
    max_depth = 100.0  # in meters, clamp very far triangulated points
    result_dir: str = os.path.join('./results', dataset_type, seq)


class CoordTransform:
    T_cv_to_ros = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])


class Visualizer:
    def __init__(self, config=None):
        self.config = config
        self.server = viser.ViserServer()

        self.server.scene.add_frame(
            name="/world_frame",
            axes_length = 5,
            axes_radius = 0.1,
        )

        self.batched_xyz = []
        self.batched_wxyz = []
        if self.config and self.config.load_gt_pose:
            self.batched_xyz_gt = []
            self.batched_wxyz_gt = []
    

    def get_pcd_handle(self, pcd_name):
        return self.server.scene.add_point_cloud(
            name=f"/world_frame/{pcd_name}",
            points=np.empty((0, 3)),
            colors=np.empty((0, 3)),
            point_size=0.005,
        )
    

    def add_cam_frame(self, T_wc, frame_names, frame_idx):
        # add camera coord frame
        tx, ty, tz = T_wc[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(T_wc[:3, :3]).as_quat()
        self.batched_xyz.append((tx, ty, tz))
        self.batched_wxyz.append((qw, qx, qy, qz))
        self.server.scene.add_batched_axes(
            f"/world_frame/{frame_names}/cam_{frame_idx}",
            axes_length = 1,
            axes_radius = 0.1,
            batched_wxyzs=self.batched_wxyz,
            batched_positions=self.batched_xyz,
        )
    

    def add_cam_frustum(self, curr_img, K, T_wc, frustum_names, frame_idx):
        # convert to RGB only for Viser display
        curr_img_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)

        # add camera frustrum
        fov = 2 * np.arctan2(curr_img_rgb.shape[0] / 2, K[0, 0])
        aspect = curr_img_rgb.shape[1] / curr_img_rgb.shape[0]
        downsample_factor = 1
        self.server.scene.add_camera_frustum(
            f"/world_frame/{frustum_names}/cam_{frame_idx}",
            fov=fov,
            aspect=aspect,
            scale=1,
            image=curr_img_rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(T_wc[:3, :3]).wxyz,
            position=T_wc[:3, 3],
        )


    def add_gt_frame(self, curr_T_gt):
        T_viz_gt = CoordTransform.T_cv_to_ros @ curr_T_gt @ np.eye(4).T
        tx_gt, ty_gt, tz_gt = T_viz_gt[:3, 3]
        qx_gt, qy_gt, qz_gt, qw_gt = Rotation.from_matrix(T_viz_gt[:3, :3]).as_quat()
        self.batched_xyz_gt.append((tx_gt, ty_gt, tz_gt))
        self.batched_wxyz_gt.append((qw_gt, qx_gt, qy_gt, qz_gt))
        self.server.scene.add_batched_axes(
            "/world_frame/ground_truth_frames",
            axes_length = 1.5,
            axes_radius = 0.15,
            batched_wxyzs=self.batched_wxyz_gt,
            batched_positions=self.batched_xyz_gt,
        )


class MonoVO:
    def __init__(self, config):
        self.config = config
        
        # dataloader
        loader_classes = {
            'KITTI': KITTILoader,
            'EuRoC': EuRoCLoader,
        }
        try:
            loader = loader_classes[config.dataset_type](config)
        except KeyError:
            raise ValueError(f"Unsupported dataset type: {config.dataset_type}")
        
        # load data
        self.img_files_list = loader.load_img_files()
        self.num_frames = len(self.img_files_list)
        cam_config = loader.load_intrinsics()
        self.K = np.array([[cam_config['intrinsics']['fx'], 0, cam_config['intrinsics']['cx']],
                           [0, cam_config['intrinsics']['fy'], cam_config['intrinsics']['cy']],
                           [0, 0, 1]])
        if self.config.load_gt_pose:
            self.gt_pose = loader.load_gt_pose()

        # predicted poses and points
        self.T_wc_list = []
        self.pcd_all = []
        self.pcd_colors_all = []
        
        # output pose file
        output_dir = os.path.join(self.config.result_dir, self.config.dataset_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.out_pose_file = os.path.join(output_dir, self.config.seq + '-traj_est.txt')
        
        # initialize visualizer
        self.visualizer = Visualizer(self.config)
        self.pcd_handle = self.visualizer.get_pcd_handle("point_cloud")


    def extract_match_features(self, prev_img, curr_img, n_feats=10000, matcher_type='bf', show_matches=True):
        """ Use ORB Feature to do feature matching """
        # create ORB features
        orb = cv2.ORB_create(nfeatures=n_feats)

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(prev_img, None)
        kp2, des2 = orb.detectAndCompute(curr_img, None)

        if matcher_type == 'bf':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif matcher_type == 'flann':
            FLANN_INDEX_LSH = 6
            matcher = cv2.FlannBasedMatcher(
                dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1),
                dict(checks=50))

        # Match ORB descriptors
        matches = matcher.match(des1, des2)

        # Sort the matched keypoints in the order of matching distance
        # so the best matches came to the front
        matches = sorted(matches, key=lambda x: x.distance)

        if show_matches:
            img_matching = cv2.drawMatches(prev_img, kp1, curr_img, kp2, matches[0:100], None)
            cv2.imshow('feature matching', img_matching)
            cv2.waitKey(0)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        return pts1, pts2


    def compute_pose(self, pts1, pts2):
        # compute essential matrix
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        E, mask = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=1)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        _, R_cw, t_cw, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx, cy))
        T_cw = np.eye(4)
        T_cw[0:3, 0:3] = R_cw
        T_cw[0:3, 3] = t_cw.flatten()

        return np.linalg.inv(T_cw)


    def triangulate(self, kp1, kp2, K, T_1w, T_2w):
        """
        Triangulate 3D points given two camera poses and keep only points in front of both cameras
        """
        # Compute projection matrices
        P1 = K @ np.linalg.inv(T_1w)[:3, :]  # Intrinsics * Extrinsics
        P2 = K @ np.linalg.inv(T_2w)[:3, :]

        # Perform triangulation
        X_homo = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)

        # Convert from homogeneous to 3D
        X = X_homo[:3] / X_homo[3]

        # Cheirality check in both camera frames
        ones = np.ones((1, X.shape[1]))
        X_h = np.vstack((X, ones))
        T1_inv = np.linalg.inv(T_1w)
        T2_inv = np.linalg.inv(T_2w)
        X_cam1 = T1_inv[:3, :4] @ X_h
        X_cam2 = T2_inv[:3, :4] @ X_h
        valid_mask = (X_cam1[2, :] > 0) & (X_cam2[2, :] > 0)

        # Optional depth clamp in second camera frame
        if self.config.max_depth is not None:
            depth = X_cam2[2, :]
            valid_mask = valid_mask & (depth < self.config.max_depth)

        return X[:, valid_mask], valid_mask
    

    def save_output_pose(self, frame_idx, T_wc):
        curr_R_wc = T_wc[:3, :3]
        curr_t_wc = T_wc[:3, 3].reshape(3, 1)
        [qx, qy, qz, qw] = Rotation.from_matrix(curr_R_wc).as_quat()        
        with open(self.out_pose_file, 'a') as f:
            f.write('%f %f %f %f %f %f %f %f\n' % (frame_idx, 
                                                    curr_t_wc.flatten()[0], 
                                                    curr_t_wc.flatten()[1], 
                                                    curr_t_wc.flatten()[2],
                                                    qx, qy, qz, qw))
    

    def mapping_points(self, pts, pcd, curr_img):
        """ 
        get 3D points and valid mask and register points in world frame (ROS convention)
        """
        # sample RGB colors for the matched pixels in the current image and register points
        u = np.clip(pts[:, 0].astype(int), 0, curr_img.shape[1]-1)
        v = np.clip(pts[:, 1].astype(int), 0, curr_img.shape[0]-1)
        colors_rgb = curr_img[v, u, ::-1] / 255.0  # BGR->RGB to [0,1]

        pcd = np.concatenate((pcd, np.ones((1, pcd.shape[1]))), axis=0)
        pcd = CoordTransform.T_cv_to_ros @ pcd
        pcd = pcd[:3, :].T

        self.pcd_all.extend(pcd.tolist())
        self.pcd_colors_all.extend(colors_rgb.tolist())


    def meet_keyframe_criteria(self, pts1, pts2, T_cc, min_parallax=1.0, min_translation=0.1):
        """Select keyframes based on a few criteria"""

        # enough baseline for good triangulation
        translation = np.linalg.norm(T_cc[:3, 3])
        if translation < min_translation:
            enough_baseline = False
        else:
            enough_baseline = True
        
        # sufficient parallax between matched points
        median_parallax = np.median(np.linalg.norm(pts2 - pts1, axis=1))
        if median_parallax < min_parallax:
            sufficient_parallax = False  
        else:
            sufficient_parallax = True
                
        return enough_baseline and sufficient_parallax


    def run(self):
        for frame_idx in range(self.num_frames):
            curr_img = cv2.imread(self.img_files_list[frame_idx], 1)
            if self.config.load_gt_pose:
                curr_T_gt = self.gt_pose[frame_idx]

            if frame_idx == 0:
                curr_T_wc = np.eye(4)
                curr_T_wc_ros = CoordTransform.T_cv_to_ros @ curr_T_wc @ np.eye(4).T
                is_kf = True
            else:
                prev_img = cv2.imread(self.img_files_list[frame_idx-1], 0)

                # feature extraction and matching, then compute relative pose in cam frame 
                prev_kps, curr_kps = self.extract_match_features(prev_img, curr_img, matcher_type='flann', show_matches=False)
                T_cc = self.compute_pose(prev_kps, curr_kps)

                # get current pose
                curr_T_wc = prev_T_wc @ T_cc
                self.T_wc_list.append(curr_T_wc)
                curr_T_wc_ros = CoordTransform.T_cv_to_ros @ curr_T_wc @ np.eye(4).T
                        
                # triangulate only in key frames
                is_kf = self.meet_keyframe_criteria(prev_kps, curr_kps, T_cc)
                if is_kf:
                    # get 3D points and valid mask
                    pcd, valid_mask = self.triangulate(prev_kps, curr_kps, self.K, prev_T_wc, curr_T_wc)
                    if not np.any(valid_mask):
                        continue
                    curr_kps = curr_kps[valid_mask]
                    self.mapping_points(curr_kps, pcd, curr_img)

                    # update point cloud handle (avoids re-adding every time)
                    self.pcd_handle.points = np.array(self.pcd_all)
                    self.pcd_handle.colors = np.array(self.pcd_colors_all)

                # draw the current image with keypoints
                keypoints = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in curr_kps]
                curr_img_kp = cv2.drawKeypoints(curr_img, keypoints, None, color=(0, 255, 0), flags=0)
                cv2.imshow('keypoints from current image', curr_img_kp)
                cv2.waitKey(1)

            # save current trajectory
            self.save_output_pose(frame_idx, curr_T_wc_ros)

            # update prev pose
            prev_T_wc = curr_T_wc
            if self.config.load_gt_pose:
                prev_T_gt = curr_T_gt

            # add camera frames and frustum for visualization
            if is_kf:
                self.visualizer.add_cam_frame(curr_T_wc_ros, "cam_frames", frame_idx)
                self.visualizer.add_cam_frustum(curr_img, self.K, curr_T_wc_ros, "cam_frustum", frame_idx)
                if config.load_gt_pose:
                    self.visualizer.add_gt_frame(curr_T_gt)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = tyro.cli(Config)
    MonoVO(config).run()
    input("Press Enter to stop...")