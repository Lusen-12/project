#include "camera.h"

bool Camera::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开相机参数文件: " << path << std::endl;
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix_;
    fs["distortion_coefficients"] >> distCoeffs_;

    if (cameraMatrix_.empty()) {
        std::cerr << "相机内参为空" << std::endl;
        return false;
    }
    return true;
}
