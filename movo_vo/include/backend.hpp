#pragma once

#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>


struct ReprojectionError 
{
    ReprojectionError(double obs_u, double obs_v,
                      double fx, double fy, double cx, double cy)
        : obs_u_(obs_u), obs_v_(obs_v),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy){}

    template <typename T>
    bool operator()(const T* const se3_params,
                    const T* const point,
                    T* residuals) const 
    {
        Eigen::Map<Sophus::SE3<T> const> T_cw(se3_params);
        Eigen::Matrix<T, 3, 1> X_w(point[0], point[1], point[2]);
        Eigen::Matrix<T, 3, 1> X_c = T_cw * X_w;

        T inv_z = T(1.0) / X_c[2];
        T u_proj = T(fx_) * X_c[0] * inv_z + T(cx_);
        T v_proj = T(fy_) * X_c[1] * inv_z + T(cy_);

        residuals[0] = u_proj - T(obs_u_);
        residuals[1] = v_proj - T(obs_v_);
        return true;
    }

    static ceres::CostFunction* Create(double obs_u, double obs_v,
                                        double fx, double fy,
                                        double cx, double cy) 
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, Sophus::SE3d::num_parameters, 3>
        (
            new ReprojectionError(obs_u, obs_v, fx, fy, cx, cy)
        );
    }

    double obs_u_, obs_v_;
    double fx_, fy_, cx_, cy_;
};


inline void run_ba(std::vector<Eigen::Matrix4d>& kf_poses,
                const std::vector<Eigen::MatrixX2d>& kf_keypoints,
                const std::vector<std::vector<int>>& kf_point_ids,
                std::vector<Eigen::Vector3d>& points_3d,
                const Eigen::Matrix3d& K,
                int steps = 50)
{
    const int num_kf = static_cast<int>(kf_poses.size());
    const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

    // Convert poses to SE3 (T_cw for projection)
    std::vector<Sophus::SE3d> cam_poses(num_kf);
    for (int i = 0; i < num_kf; ++i) 
    {
        cam_poses[i] = Sophus::SE3d(kf_poses[i].inverse());
    }

    // Copy points to contiguous array
    std::vector<std::array<double, 3>> pt_params(points_3d.size());
    for (size_t i = 0; i < points_3d.size(); ++i) {
        pt_params[i] = {points_3d[i](0), points_3d[i](1), points_3d[i](2)};
    }

    // Build Ceres problem
    ceres::Problem problem;
    ceres::Manifold* se3_manifold = new Sophus::Manifold<Sophus::SE3>();

    for (int k = 0; k < num_kf; ++k) 
    {
        problem.AddParameterBlock(cam_poses[k].data(), Sophus::SE3d::num_parameters, se3_manifold);
    }

    for (int k = 0; k < num_kf; ++k) 
    {
        for (int j = 0; j < static_cast<int>(kf_point_ids[k].size()); ++j) 
        {
            int pid = kf_point_ids[k][j];
            double u = kf_keypoints[k](j, 0);
            double v = kf_keypoints[k](j, 1);

            ceres::CostFunction* cost = ReprojectionError::Create(u, v, fx, fy, cx, cy);
            ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(cost, loss, cam_poses[k].data(), pt_params[pid].data());
        }
    }

    problem.SetParameterBlockConstant(cam_poses[0].data());

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = steps;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Print loss
    int n_obs = 0;
    for (int k = 0; k < num_kf; ++k) 
    {
        n_obs += static_cast<int>(kf_point_ids[k].size());
    }
    double rms_error = std::sqrt(summary.final_cost / n_obs);
    std::cout << "[BA] final RMS reprojection error: " << rms_error << " px" << std::endl;

    // Write back
    for (int i = 0; i < num_kf; ++i) 
    {
        kf_poses[i] = cam_poses[i].matrix().inverse();
    }
    for (size_t i = 0; i < points_3d.size(); ++i) 
    {
        points_3d[i] = Eigen::Vector3d(pt_params[i][0], pt_params[i][1], pt_params[i][2]);
    }
}