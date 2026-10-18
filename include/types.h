#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// 单个装甲板检测结果
struct Armor {
    cv::Rect box;                       // 装甲板矩形框
    std::vector<cv::Point2f> corners;   // 四个角点（顺序：左上、右上、右下、左下）
    int digit = -1;                     // 数字 1~9，-1 表示未识别
    float confidence = 0.0f;            // 检测置信度
    int classId = -1;                   // 类别索引
};

// 位置解算结果
struct PoseResult {
    bool success = false;
    cv::Vec3d tvec;   // 相机坐标系位置 (x, y, z)，单位 mm
    cv::Vec3d rvec;   // 旋转向量
};

// 误差结果
struct ErrorResult {
    std::string imageName;
    cv::Vec3d calc;    // 计算值
    cv::Vec3d truth;   // 真值
    cv::Vec3d delta;   // calc - truth
    double error3d;    // 三维误差
};
