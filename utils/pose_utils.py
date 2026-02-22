import math
import numpy as np
from scipy.spatial.transform import Rotation


def read_file_list(filename):
    """Reads a trajectory from a text file (TUM format).

    File format: "stamp tx ty tz qx qy qz qw" per line.

    Returns:
        list of (stamp, [tx,ty,tz,qx,qy,qz,qw]) sorted by stamp
    """
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    parsed = [[v.strip() for v in line.split(" ") if v.strip() != ""]
              for line in lines if len(line) > 0 and line[0] != "#"]
    parsed = [(float(l[0]), l[1:]) for l in parsed if len(l) > 1]
    parsed.sort(key=lambda x: x[0])
    return parsed


def _to_mat(entry):
    """Convert [tx,ty,tz,qx,qy,qz,qw] strings to 4x4 matrix."""
    vals = [float(v) for v in entry]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(vals[3:7]).as_matrix()
    T[:3, 3] = vals[0:3]
    return T


def compute_ate(gtruth_file, pred_file):
    """Compute Absolute Trajectory Error (RMSE after scale+translation alignment).

    GT and prediction files must have the same frame stamps (matched 1-to-1).
    """
    gt_list = read_file_list(gtruth_file)
    pr_list = read_file_list(pred_file)
    n = min(len(gt_list), len(pr_list))
    if n < 2:
        return False

    gt_xyz = np.array([[float(v) for v in gt_list[i][1][0:3]] for i in range(n)])
    pr_xyz = np.array([[float(v) for v in pr_list[i][1][0:3]] for i in range(n)])

    # align first frame
    offset = gt_xyz[0] - pr_xyz[0]
    pr_xyz += offset[None, :]

    # optimal scale
    scale = np.sum(gt_xyz * pr_xyz) / np.sum(pr_xyz ** 2)
    alignment_error = pr_xyz * scale - gt_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / n
    return rmse


def compute_rpe(gtruth_file, pred_file, delta=1):
    """Compute Relative Pose Error (translational RMSE in m, rotational RMSE in deg).

    GT and prediction files must have the same frame stamps (matched 1-to-1).
    """
    gt_list = read_file_list(gtruth_file)
    pr_list = read_file_list(pred_file)
    n = min(len(gt_list), len(pr_list))
    if n < delta + 1:
        return False, False

    gt_poses = [_to_mat(gt_list[i][1]) for i in range(n)]
    pr_poses = [_to_mat(pr_list[i][1]) for i in range(n)]

    trans_errors = []
    rot_errors = []
    for i in range(n - delta):
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        pr_rel = np.linalg.inv(pr_poses[i]) @ pr_poses[i + delta]

        E = np.linalg.inv(gt_rel) @ pr_rel

        trans_errors.append(np.linalg.norm(E[:3, 3]))

        cos_angle = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors.append(math.degrees(math.acos(cos_angle)))

    rpe_trans = math.sqrt(np.mean(np.array(trans_errors) ** 2))
    rpe_rot = math.sqrt(np.mean(np.array(rot_errors) ** 2))
    return rpe_trans, rpe_rot
