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


class Frame:
    def __init__(self, rgb, K, T_wc):
        self.rgb = rgb
        self.K = K
        # camera pose represented in world frame
        self.T_wc = T_wc


class MonoVO:
    def __init__(self):
        # dataloader
        config = Config()
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
        self.gt_pose = loader.load_gt_pose()
        
        # output pose file
        output_dir = os.path.join(config.result_dir, config.dataset_type)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.out_pose_file = os.path.join(output_dir, config.seq + '-traj_est.txt')


    def extractAndMatchFeature(self, prev_img, curr_img, n_feats=10000, show_matches=True):
        """ Use ORB Feature to do feature matching """
        # create ORB features
        orb = cv2.ORB_create(nfeatures=n_feats)

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(prev_img, None)
        kp2, des2 = orb.detectAndCompute(curr_img, None)

        # use brute-force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match ORB descriptors
        matches = bf.match(des1, des2)

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
        """Triangulate 3D points given two camera poses and keep only points in front of both cameras."""
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
        if config.max_depth is not None:
            depth = X_cam2[2, :]
            valid_mask = valid_mask & (depth < config.max_depth)

        return X[:, valid_mask], valid_mask


    def run(self):
        server = viser.ViserServer()
    
        T_wc_list = []
        batched_xyz = []
        batched_wxyz = []
        pcd_all = []
        pcd_colors_all = []
        pcd_handle = server.scene.add_point_cloud(
            name="/vision_frame/point_cloud",
            points=np.empty((0, 3)),
            colors=np.empty((0, 3)),
            point_size=0.005,
        )
        for i in range(self.num_frames):
            curr_img = cv2.imread(self.img_files_list[i], 1)
            curr_T_gt = self.gt_pose[i]

            if i == 0:
                curr_T_wc = np.eye(4)
            else:
                # prev_imgName = img_data_dir + str(i - 1).zfill(6) + '.png'
                prev_img = cv2.imread(self.img_files_list[i-1], 0)

                # feature extraction and matching, then compute relative pose in cam frame 
                pts1, pts2 = self.extractAndMatchFeature(prev_img, curr_img, show_matches=False)
                T_cc = self.compute_pose(pts1, pts2)

                # get current pose
                curr_T_wc = prev_T_wc @ T_cc
                T_wc_list.append(curr_T_wc)
                        
                # triangulate only in key frames
                if i%config.key_frame_interval == 0:
                    pcd, valid_mask = self.triangulate(pts1, pts2, self.K, prev_T_gt, curr_T_gt)
                    if not np.any(valid_mask):
                        continue
                    pts2 = pts2[valid_mask]

                    # sample RGB colors for the matched pixels in the current image
                    u = np.clip(pts2[:, 0].astype(int), 0, curr_img.shape[1]-1)
                    v = np.clip(pts2[:, 1].astype(int), 0, curr_img.shape[0]-1)
                    colors_rgb = curr_img[v, u, ::-1] / 255.0  # BGR->RGB to [0,1]

                    pcd = np.concatenate((pcd, np.ones((1, pcd.shape[1]))), axis=0)
                    pcd = CoordTransform.T_cv_to_ros @ pcd
                    pcd = pcd[:3, :].T

                    pcd_all.extend(pcd.tolist())
                    pcd_colors_all.extend(colors_rgb.tolist())

                    # update point cloud handle (avoids re-adding every time)
                    pcd_handle.points = np.array(pcd_all)
                    pcd_handle.colors = np.array(pcd_colors_all)

                # draw the current image with keypoints
                keypoints = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in pts2]
                curr_img_kp = cv2.drawKeypoints(curr_img, keypoints, None, color=(0, 255, 0), flags=0)
                cv2.imshow('keypoints from current image', curr_img_kp)
                cv2.waitKey(1)

            # save current trajectory
            curr_R_wc = curr_T_wc[:3, :3]
            curr_t_wc = curr_T_wc[:3, 3].reshape(3, 1)
            [qx, qy, qz, qw] = Rotation.from_matrix(curr_R_wc).as_quat()        
            with open(self.out_pose_file, 'a') as f:
                f.write('%f %f %f %f %f %f %f %f\n' % (i, 
                                                       curr_t_wc.flatten()[0], 
                                                       curr_t_wc.flatten()[1], 
                                                       curr_t_wc.flatten()[2],
                                                       qx, qy, qz, qw))

            # update prev pose
            prev_T_wc = curr_T_wc
            prev_T_gt = curr_T_gt

            # add camera coord frame
            if i % config.key_frame_interval == 0:
                # T_viz = CoordTransform.T_cv_to_ros @ curr_T_wc @ np.eye(4).T
                T_viz = CoordTransform.T_cv_to_ros @ curr_T_gt @ np.eye(4).T
                tx, ty, tz = T_viz[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(T_viz[:3, :3]).as_quat()
                batched_xyz.append((tx, ty, tz))
                batched_wxyz.append((qw, qx, qy, qz))
                server.scene.add_batched_axes(
                    "/vision_frame",
                    axes_length = 1,
                    axes_radius = 0.1,
                    batched_wxyzs=batched_wxyz,
                    batched_positions=batched_xyz,
                )

                # convert to RGB only for Viser display
                curr_img_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)

                # define current frame
                frame = Frame(
                    rgb=curr_img_rgb,
                    K = self.K,
                    T_wc = T_viz,
                )

                # add camera frustrum
                fov = 2 * np.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
                aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
                downsample_factor = 1
                server.scene.add_camera_frustum(
                    f"/vision_frame/cam_frustum/cam_{i}",
                    fov=fov,
                    aspect=aspect,
                    scale=1,
                    image=frame.rgb[::downsample_factor, ::downsample_factor],
                    wxyz=tf.SO3.from_matrix(frame.T_wc[:3, :3]).wxyz,
                    position=frame.T_wc[:3, 3],
                )

        cv2.destroyAllWindows()


if __name__ == "__main__":
    config = tyro.cli(Config)
    MonoVO().run()
    while True:
        time.sleep(1)