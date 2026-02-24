import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import viser
import viser.transforms as tf
sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.mono_vo import MonoVO
vo = MonoVO()

img1 = cv2.imread('./tests/1.png')
img2 = cv2.imread('./tests/2.png')
pts1, pts2 = vo.extractAndMatchFeature(img1, img2, show_matches=False)
T = vo.compute_pose(pts1, pts2)
K = np.array([[718.856, 0, 607.1928],
              [0, 718.856, 185.2157],
              [0, 0, 1]])
pts_3d_w, _, _ = vo.triangulate(pts1, pts2, K, np.eye(4), T)

pts_int = np.round(pts1).astype(int)
h, w, _ = img1.shape
valid_mask = (pts_int[:, 0] >= 0) & (pts_int[:, 0] < w) & \
                (pts_int[:, 1] >= 0) & (pts_int[:, 1] < h)
pts_3d_w = pts_3d_w[:, valid_mask]
pts_int = pts_int[valid_mask]
colors_bgr = img1[pts_int[:, 1], pts_int[:, 0]]
colors_rgb = colors_bgr[:, ::-1]
pts_3d_w = pts_3d_w.T

# Start viser server
server = viser.ViserServer()

# Add point cloud
server.scene.add_point_cloud(
    name="my_pointcloud",
    points=pts_3d_w,
    colors = colors_rgb,
    point_size=0.01
)

class Frame:
    def __init__(self, rgb, K, T_wc):
        self.rgb = rgb
        self.K = K
        # camera pose represented in world frame
        self.T_wc = T_wc

img = [img1, img2]
Ts = [np.eye(4), T]
batched_xyz = []
batched_wxyz = []

for i in range(2):
    # define current frame
    frame = Frame(
        rgb=img[i],
        K = K,
        T_wc = Ts[i]
    )

    tx, ty, tz = Ts[i][:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(Ts[i][:3, :3]).as_quat()
    batched_xyz.append((tx, ty, tz))
    batched_wxyz.append((qw, qx, qy, qz))
    server.scene.add_batched_axes(
        "/tree",
        axes_length = 0.2,
        axes_radius = 0.01,
        batched_wxyzs=batched_wxyz,
        batched_positions=batched_xyz,
    )

    # add camera frustrum
    fov = 2 * np.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
    aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
    downsample_factor = 1
    server.scene.add_camera_frustum(
        f"/frames/t{i}/frustum",
        fov=fov,
        aspect=aspect,
        scale=0.2,
        image=frame.rgb[::downsample_factor, ::downsample_factor],
        wxyz=tf.SO3.from_matrix(frame.T_wc[:3, :3]).wxyz,
        position=frame.T_wc[:3, 3],
    )

print("Open http://localhost:8080 to view the point cloud.")

# Keep server running
while True:
    time.sleep(1)