#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>

class DetectorTraditional {
public:
    std::vector<Armor> detect(const cv::Mat& image);
};
