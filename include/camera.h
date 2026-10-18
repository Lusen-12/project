#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class Camera {
public:
    // 从 yaml 文件加载内参和畸变系数
    bool load(const std::string& path);

    cv::Mat getMatrix() const { return cameraMatrix_; }
    cv::Mat getDistCoeffs() const { return distCoeffs_; }

private:
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
};
