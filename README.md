# VSLAM Minimum Implementation

This repository contains the minimum implementation of VSLAM, including:

## Monocular Visual Odometry in Python

This is a simple implementation of monocular visual odometry in Python.

### env deployment with venv or conda
#### venv
```bash
python -m venv py_vo
source py_vo/bin/activate
pip install -r requirements.txt
```
#### conda
```bash
conda create -n py_vo python=3.10
conda activate py_vo
pip install -r requirements.txt
```

### download datasets
#### KITTI Odometry
<img src="./mono_vo_py/assets/kitti_09.png" alt="KITTI Odometry sequence 09" width="60%" />

#### EuRoC MAV Dataset (TODO)


### run the code
```bash
python mono_vo.py --dataset_type KITTI --seq 09
```

## Visual SLAM minimum implementation