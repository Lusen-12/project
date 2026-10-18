#include "visualizer.h"

void Visualizer::drawArmor(cv::Mat& image, const Armor& armor) {
    cv::rectangle(image, armor.box, cv::Scalar(0, 255, 0), 2);
    if (armor.digit > 0) {
        cv::putText(image, std::to_string(armor.digit),
                    cv::Point(armor.box.x, armor.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);
    }
}

void Visualizer::drawPose(cv::Mat& image, const Armor& armor, const PoseResult& pose) {
    // 1. 画绿色四边形框
    std::vector<cv::Point> intCorners;
    for (auto& p : armor.corners) intCorners.push_back(cv::Point(p.x, p.y));
    cv::polylines(image, intCorners, true, cv::Scalar(0, 255, 0), 2);

    // 2. 画黄色角点
    for (auto& p : armor.corners) {
        cv::circle(image, p, 3, cv::Scalar(0, 255, 255), -1);
    }

    // 3. 显示位置信息 t=(x, y, z)
    if (pose.success) {
        char buf[128];
        // z 除以 1000 转为米显示
        snprintf(buf, sizeof(buf), "t=(%.1f, %.1f, %.1f)", 
                 pose.tvec[0], pose.tvec[1], pose.tvec[2] / 1000.0);
        
        // 将文字画在左下角点附近
        int textY = std::max(20, intCorners[3].y + 20);
        cv::putText(image, buf, cv::Point(intCorners[3].x, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
    } else {
        cv::putText(image, "PnP Failed", cv::Point(intCorners[3].x, intCorners[3].y + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
}
