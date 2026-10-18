#include "detector_nn.h"
#include <algorithm>

DetectorNN::DetectorNN(const std::string& modelPath) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    inputName_ = session_->GetInputNameAllocated(0, allocator).get();
    outputName_ = session_->GetOutputNameAllocated(0, allocator).get();
}

cv::Mat DetectorNN::letterbox(const cv::Mat& src, cv::Size targetSize,
                              float& scale, int& padX, int& padY) {
    int w = src.cols, h = src.rows;
    scale = std::min(targetSize.width * 1.0f / w, targetSize.height * 1.0f / h);
    int newW = int(w * scale);
    int newH = int(h * scale);
    padX = (targetSize.width - newW) / 2;
    padY = (targetSize.height - newH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    cv::Mat out(targetSize.height, targetSize.width, CV_8UC3,
                cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(padX, padY, newW, newH)));
    return out;
}

std::vector<Armor> DetectorNN::detect(const cv::Mat& image) {
    std::vector<Armor> armors;
    if (image.empty()) return armors;

    // 1. letterbox
    float scale;
    int padX, padY;
    cv::Mat input = letterbox(image, cv::Size(inputW, inputH), scale, padX, padY);

    // 2. BGR -> RGB
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    // 3. 归一化 + NCHW
    cv::Mat f;
    input.convertTo(f, CV_32F, 1.0 / 255.0);
    std::vector<float> inputData(3 * inputH * inputW);
    for (int c = 0; c < 3; ++c)
        for (int h = 0; h < inputH; ++h)
            for (int w = 0; w < inputW; ++w)
                inputData[c * inputH * inputW + h * inputW + w] =
                    f.at<cv::Vec3f>(h, w)[c];

    std::vector<int64_t> shape = {1, 3, inputH, inputW};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(), shape.data(), shape.size());

    const char* inNames[] = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                 inNames, &tensor, 1, outNames, 1);

        // 4. 解析输出 [1, 21, 6300]
    float* out = outputs[0].GetTensorMutableData<float>();
    int numAnchors = 6300;
    int stride = 4 + numClasses;  // 21

    for (int i = 0; i < numAnchors; ++i) {
        // 按连续内存直接读取，每次步进 21
        float cx = out[i * stride + 0];
        float cy = out[i * stride + 1];
        float w  = out[i * stride + 2];
        float h  = out[i * stride + 3];

        // 类别分数
        int bestCls = -1;
        float bestScore = 0;
        for (int c = 0; c < numClasses; ++c) {
            float s = out[i * stride + 4 + c];
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }
        if (bestScore > 0.1) { 
            printf("发现目标! 类别: %d, 置信度: %.2f, 坐标: %f %f %f %f\n", bestCls, bestScore, cx, cy, w, h); 
        }
        if (bestScore < confThreshold) continue;

        // 映射回 letterbox 图
        float x1 = (cx - w / 2) * inputW;
        float y1 = (cy - h / 2) * inputH;
        float x2 = (cx + w / 2) * inputW;
        float y2 = (cy + h / 2) * inputH;

        // 映射回原图
        x1 = (x1 - padX) / scale;
        y1 = (y1 - padY) / scale;
        x2 = (x2 - padX) / scale;
        y2 = (y2 - padY) / scale;

        x1 = std::max(0.0f, std::min(x1, (float)image.cols - 1));
        y1 = std::max(0.0f, std::min(y1, (float)image.rows - 1));
        x2 = std::max(0.0f, std::min(x2, (float)image.cols - 1));
        y2 = std::max(0.0f, std::min(y2, (float)image.rows - 1));

        Armor a;
        a.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        a.confidence = bestScore;
        a.classId = bestCls;
        a.corners = {
            cv::Point2f(x1, y1), cv::Point2f(x2, y1),
            cv::Point2f(x2, y2), cv::Point2f(x1, y2)
        };
        armors.push_back(a);
    }

    // 5. NMS
    nms(armors, 0.45f);
    return armors;
}

void DetectorNN::nms(std::vector<Armor>& armors, float iouThreshold) {
    std::sort(armors.begin(), armors.end(),
              [](const Armor& a, const Armor& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Armor> keep;
    std::vector<bool> removed(armors.size(), false);

    for (size_t i = 0; i < armors.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(armors[i]);
        for (size_t j = i + 1; j < armors.size(); ++j) {
            if (removed[j]) continue;
            cv::Rect inter = armors[i].box & armors[j].box;
            float interArea = inter.area();
            float unionArea = armors[i].box.area() + armors[j].box.area() - interArea;
            float iou = interArea / unionArea;
            if (iou > iouThreshold) removed[j] = true;
        }
    }
    armors = keep;
}
