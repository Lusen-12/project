#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>

class Visualizer {
public:
    static void drawArmor(cv::Mat& image, const Armor& armor);
    static void drawPose(cv::Mat& image, const Armor& armor, const PoseResult& pose);
};
