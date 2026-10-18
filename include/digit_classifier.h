#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

class DigitClassifier {
public:
    explicit DigitClassifier(const std::string& modelPath);

    // 输入数字区域，返回 1~9，失败返回 -1
    int classify(const cv::Mat& digitROI, float* outConfidence = nullptr);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "digit"};
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_;
    std::string outputName_;
};
