#pragma once
#include "types.h"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

class DetectorNN {
public:
    explicit DetectorNN(const std::string& modelPath);

    // 输入原图，输出装甲板列表
    std::vector<Armor> detect(const cv::Mat& image);

private:
    // letterbox 缩放
    cv::Mat letterbox(const cv::Mat& src, cv::Size targetSize,
                      float& scale, int& padX, int& padY);

    // NMS
    void nms(std::vector<Armor>& armors, float iouThreshold);

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "armor"};
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;

    const int inputW = 640;
    const int inputH = 480;
    const int numClasses = 17;
    const float confThreshold = 0.25f;
};
