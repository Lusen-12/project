#include "camera.h"
#include "pose_solver.h"
#include "detector_nn.h"
#include "detector_traditional.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

int main(int argc, char** argv) {
    std::string mode = "traditional";
    if (argc > 1) mode = argv[1];

    Camera cam;
    if (!cam.load("../config/camera.yaml")) return -1;

    // 尺寸单位：米 (m)
    float armorWidth = 0.135f;
    float armorHeight = 0.055f;
    PoseSolver solver(cam, armorWidth, armorHeight);

    DetectorTraditional tradDetector;
    DetectorNN nnDetector("../models/infantry.onnx"); 

    std::ifstream truthFile("../data/images/truth.txt");
    if (!truthFile.is_open()) {
        std::cerr << "无法打开真值文件！请检查 ../data/images/truth.txt" << std::endl;
        return -1;
    }

    // 动态生成不同模式对应的 CSV 文件名
    std::string csvPath = "../output/eval_" + mode + ".csv";
    std::ofstream csvFile(csvPath);
    csvFile << "ImageName,Calc_X,Calc_Y,Calc_Z,Truth_X,Truth_Y,Truth_Z,dX,dY,dZ,Error3D\n";

    std::string line;
    std::cout << "开始处理图片，模式: " << mode << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    while (std::getline(truthFile, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string imgName;
        double tx, ty, tz;
        iss >> imgName >> tx >> ty >> tz;

        std::string imgPath = "../data/images/" + imgName;
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            std::cerr << "无法读取图片: " << imgPath << std::endl;
            continue;
        }

        std::vector<Armor> armors;
        
        // ================= 这里加入了 try-catch =================
        if (mode == "nn") {
            try {
                armors = nnDetector.detect(img);
            } catch (const std::exception& e) {
                std::cerr << "[" << imgName << "] 神经网络推理发生异常: " << e.what() << std::endl;
                // 异常后 armors 为空，程序会继续处理下一张图，不会崩溃
            }
        } else {
            armors = tradDetector.detect(img);
        }
        // ========================================================

        if (armors.empty()) {
            std::cout << "[" << imgName << "] 未识别到装甲板" << std::endl;
            continue;
        }

        // 取面积最大的装甲板
        auto bestArmor = armors[0];
        for (auto& a : armors) {
            if (a.box.area() > bestArmor.box.area()) {
                bestArmor = a;
            }
        }

        PoseResult pose = solver.solve(bestArmor.corners);

        if (!pose.success) {
            std::cout << "[" << imgName << "] PnP 解算失败" << std::endl;
            continue;
        }

        // 计算误差（单位：米）
        cv::Vec3d truth(tx, ty, tz);
        cv::Vec3d delta = pose.tvec - truth;
        double err3D = std::sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "图片: " << imgName << "\n"
                  << "  计算坐标: (" << pose.tvec[0] << ", " << pose.tvec[1] << ", " << pose.tvec[2] << ") m\n"
                  << "  真值坐标: (" << truth[0] << ", " << truth[1] << ", " << truth[2] << ") m\n"
                  << "  误差(dX, dY, dZ): (" << delta[0] << ", " << delta[1] << ", " << delta[2] << ") m\n"
                  << "  三维误差: " << err3D << " m\n" << std::endl;

        csvFile << imgName << "," 
                << pose.tvec[0] << "," << pose.tvec[1] << "," << pose.tvec[2] << ","
                << truth[0] << "," << truth[1] << "," << truth[2] << ","
                << delta[0] << "," << delta[1] << "," << delta[2] << "," << err3D << "\n";
                
        // 画图保存
        cv::rectangle(img, bestArmor.box, cv::Scalar(0,255,0), 2);
        cv::imwrite("../output/eval_" + mode + "_" + imgName, img);
    }

    std::cout << "结果已保存至 " << csvPath << std::endl;
    return 0;
}
