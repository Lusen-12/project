#include "camera.h"
#include "pose_solver.h"
#include "digit_classifier.h"
#include "detector_nn.h"
#include "detector_traditional.h"
#include "visualizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "用法: " << argv[0]
                  << " <mode: nn|traditional> <input: video|image_dir> <output>\n";
        return -1;
    }
    std::string mode = argv[1];
    std::string input = argv[2];
    std::string output = argv[3];

    // 1. 加载相机参数
    Camera cam;
    if (!cam.load("config/camera.yaml")) {
        std::cerr << "相机参数加载失败" << std::endl;
        return -1;
    }

    // 2. 位置解算器（装甲板物理尺寸，单位m )
    float armorWidth = 0.135f;
    float armorHeight = 0.055f;
    PoseSolver solver(cam, armorWidth, armorHeight);

    // 3. 数字分类器
    DigitClassifier digitCls("models/digit.onnx");

    // 4. 装甲板检测器
    DetectorNN nnDetector("models/infantry.onnx");
    DetectorTraditional tradDetector;

    // 5. 打开视频
    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频: " << input << std::endl;
        return -1;
    }

    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(w, h));
    // 检查 writer 是否成功打开
    if (!writer.isOpened()) {
        std::cerr << "错误：无法初始化视频写入器！请检查编码器。" << std::endl;
        return -1;
    }
    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<Armor> armors;
        if (mode == "nn") {
            armors = nnDetector.detect(frame);
        } else {
            armors = tradDetector.detect(frame);
        }

        if (armors.empty()) {
            cv::putText(frame, "No Armor", cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        }

        for (auto& a : armors) {
            // 1. 强制限制装甲板框在原图范围内
            a.box.x = std::max(0, a.box.x);
            a.box.y = std::max(0, a.box.y);
            a.box.width = std::min(a.box.width, frame.cols - a.box.x);
            a.box.height = std::min(a.box.height, frame.rows - a.box.y);

            // 如果框太小，直接跳过
            if (a.box.width <= 0 || a.box.height <= 0) continue;

            // 2. 计算数字识别 ROI
            cv::Rect digitROI = a.box;
	    // 装甲板数字在正中间，宽度取中间的 40%，高度取中间的 80%
            digitROI.x += digitROI.width * 0.3; 
            digitROI.width *= 0.4;
            digitROI.y += digitROI.height * 0.1;
            digitROI.height *= 0.8;
            // 再次确保 ROI 不越界
            digitROI.x = std::max(0, digitROI.x);
            digitROI.y = std::max(0, digitROI.y);
            digitROI.width = std::min(digitROI.width, frame.cols - digitROI.x);
            digitROI.height = std::min(digitROI.height, frame.rows - digitROI.y);

            int digit = -1;
            if (digitROI.width > 0 && digitROI.height > 0) {
                digit = digitCls.classify(frame(digitROI));
            }
            a.digit = digit;

            // 3. 位置解算
            PoseResult pose = solver.solve(a.corners);

            // 4. 画图
            Visualizer::drawArmor(frame, a);
            Visualizer::drawPose(frame, a, pose);
        }

        writer.write(frame);
        // 如果是在无图形界面的服务器上，cv::imshow 可能会报错，如果不需要看实时窗口，可以把下面两行注释掉
        cv::imshow("result", frame);
        if (cv::waitKey(1) == 27) break;
    } // 这个括号是 while 循环的结尾，千万别漏了！

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}
