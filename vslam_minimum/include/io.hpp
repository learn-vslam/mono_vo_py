#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fs = std::filesystem;


class Config {
public:
    std::string data_root  = "/media/" + std::string(getenv("USER")) + "/SeagateDrive/ws/datasets/";
    std::string dataset_type = "KITTI";
    bool   load_gt_pose    = true;
    std::string seq        = "09";
    int    key_frame_interval = 5;
    double max_depth       = 100.0;
    std::string result_dir; // set in constructor

    Config() 
    {
        result_dir = "./results/" + dataset_type + "/" + seq;
    }
};


class KITTILoader {
public:
    Config cfg;

    KITTILoader(const Config& c) : cfg(c) {}

    std::vector<std::string> load_img_files() 
    {
        std::string img_dir = cfg.data_root +
            "/kitti-odom/data_odometry_color/dataset/sequences/" +
            cfg.seq + "/image_2";
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(img_dir))
            if (e.path().extension() == ".png")
                files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        return files;
    }

    // Loads from config/kitti_odom.yaml (relative to cwd)
    // Falls back to hardcoded KITTI seq 00-10 intrinsics if file missing
    cv::Mat load_intrinsics() 
    {
        double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

        std::ifstream f("./config/kitti_odom.yaml");
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line)) {
                std::istringstream ss(line);
                std::string key;
                ss >> key;
                if (key == "fx:") ss >> fx;
                else if (key == "fy:") ss >> fy;
                else if (key == "cx:") ss >> cx;
                else if (key == "cy:") ss >> cy;
            }
        }

        return (cv::Mat_<double>(3,3) <<
            fx, 0, cx,
            0, fy, cy,
            0,  0,  1);
    }

    std::vector<Eigen::Matrix4d> load_gt_pose() 
    {
        std::string gt_path = cfg.data_root +
            "/kitti-odom/data_odometry_poses/dataset/poses/" +
            cfg.seq + ".txt";
        std::vector<Eigen::Matrix4d> poses;
        std::ifstream f(gt_path);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    ss >> T(r, c);
            poses.push_back(T);
        }
        return poses;
    }
};