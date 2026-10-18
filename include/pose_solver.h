#pragma once
#include "types.h"
#include "camera.h"

class PoseSolver {
public:
    // armorWidth/armorHeight: 装甲板物理尺寸，单位 mm
    PoseSolver(const Camera& cam, float armorWidth, float armorHeight);

    // imagePoints: 图像上四个角点，顺序与 objectPoints_ 一致
    PoseResult solve(const std::vector<cv::Point2f>& imagePoints) const;

    // 三维误差
    static double computeError(const cv::Vec3d& calc, const cv::Vec3d& truth);

private:
    Camera cam_;
    std::vector<cv::Point3f> objectPoints_;
};
