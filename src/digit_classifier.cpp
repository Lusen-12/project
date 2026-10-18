#include "digit_classifier.h"

DigitClassifier::DigitClassifier(const std::string& modelPath) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    inputName_ = session_->GetInputNameAllocated(0, allocator).get();
    outputName_ = session_->GetOutputNameAllocated(0, allocator).get();
}

int DigitClassifier::classify(const cv::Mat& digitROI, float* outConfidence) {
    if (digitROI.empty()) return -1;

    // 1. 转灰度
    cv::Mat gray;
    if (digitROI.channels() == 3)
        cv::cvtColor(digitROI, gray, cv::COLOR_BGR2GRAY);
    else
        gray = digitROI;

    // 2. resize 到 28x20（宽x高）
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(28, 20));

    // 3. 归一化到 0~1
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 4. 构造 1x20x28x1 张量
    std::vector<float> inputData(20 * 28);
    for (int h = 0; h < 20; ++h)
        for (int w = 0; w < 28; ++w)
            inputData[h * 28 + w] = resized.at<float>(h, w);

    std::vector<int64_t> inputShape = {1, 20, 28, 1};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size());

    const char* inputNames[] = {inputName_.c_str()};
    const char* outputNames[] = {outputName_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 inputNames, &inputTensor, 1,
                                 outputNames, 1);

    float* out = outputs[0].GetTensorMutableData<float>();
    int argmax = 0;
    float maxVal = out[0];
    for (int i = 1; i < 9; ++i) {
        if (out[i] > maxVal) { maxVal = out[i]; argmax = i; }
    }

    if (outConfidence) *outConfidence = maxVal;
    return argmax + 1;
}
