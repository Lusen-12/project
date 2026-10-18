#include "pose_solver.h"
#include <cmath>

PoseSolver::PoseSolver(const Camera& cam, float armorWidth, float armorHeight)
    : cam_(cam) {
    float w = 0.135f / 2.0f;
    float h = 0.055f / 2.0f;
    
    objectPoints_ = {
        cv::Point3f(-w, -h, 0),  // 左上
        cv::Point3f( w, -h, 0),  // 右上
        cv::Point3f( w,  h, 0),  // 右下
        cv::Point3f(-w,  h, 0)   // 左下
    };
}
PoseResult PoseSolver::solve(const std::vector<cv::Point2f>& imagePoints) const {
    PoseResult result;
    if (imagePoints.size() != 4) {
        return result;
    }

    cv::Mat rvec, tvec;
    bool ok = cv::solvePnP(objectPoints_, imagePoints,
                           cam_.getMatrix(), cam_.getDistCoeffs(),
                           rvec, tvec);
    if (!ok) {
        return result;
    }

    result.success = true;
    result.rvec = cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
    result.tvec = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    return result;
}

double PoseSolver::computeError(const cv::Vec3d& calc, const cv::Vec3d& truth) {
    cv::Vec3d d = calc - truth;
    return std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
}
