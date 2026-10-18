#include "detector_traditional.h"
#include <cmath>
#include <algorithm>

// 辅助函数：计算两个点的距离
float getDistance(cv::Point2f a, cv::Point2f b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

std::vector<Armor> DetectorTraditional::detect(const cv::Mat& image) {
    std::vector<Armor> armors;
    if (image.empty()) return armors;

    // 1. HSV 颜色分割（专门适配仿真环境的蓝紫发光灯条）
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // 提取亮度极高、且带有一点蓝色的区域
    // H(色调): 100~140 涵盖了蓝到紫
    // S(饱和度): 40~255 (仿真环境的灯条边缘饱和度低，放宽下限)
    // V(亮度): 180~255 (环境很暗，灯条极亮，提高亮度门槛排除背景噪点)
    cv::inRange(hsv, cv::Scalar(100, 40, 180), cv::Scalar(140, 255, 255), mask);

    //2. 让断开的灯条连成一条线
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 3. 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4. 筛选灯条，并保存下来
    struct LightBar {
        cv::RotatedRect rect;
        cv::Point2f center;
    };
    std::vector<LightBar> lights;

    for (const auto& c : contours) {
        cv::RotatedRect rr = cv::minAreaRect(c);
        float w = rr.size.width;
        float h = rr.size.height;
        if (w > h) std::swap(w, h); // 保证 h 是长边

        float ratio = h / w;
        float area = rr.size.area();

        if (ratio > 1.2 && ratio < 20.0 && area > 10) {
            lights.push_back({rr, rr.center});
        }
    }

    // 5. 灯条两两配对
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            cv::Point2f c1 = lights[i].center;
            cv::Point2f c2 = lights[j].center;

            // 约束 1：高度差不能太大
            if (std::abs(c1.y - c2.y) > 50) continue;

            // 约束 2：距离约束
            float dx = std::abs(c1.x - c2.x);
            if (dx < 20 || dx > 300) continue;
	    float h1 = std::max(lights[i].rect.size.width, lights[i].rect.size.height);
            float h2 = std::max(lights[j].rect.size.width, lights[j].rect.size.height);
            float h_ratio = h1 / h2;
            if (h_ratio < 0.6f || h_ratio > 1.6f) continue;

            // 约束 3：角度约束（两根灯条倾斜角度应当一致）
            float angle1 = lights[i].rect.angle;
            float angle2 = lights[j].rect.angle;
            if (std::abs(angle1 - angle2) > 15.0f && std::abs(angle1 - angle2) < 165.0f) continue;

            // 计算装甲板的四个角点（直接利用灯条的最小外接矩形）
            cv::Point2f pts1[4], pts2[4];
            lights[i].rect.points(pts1);
            lights[j].rect.points(pts2);

            // 对 4 个点进行排序，找到左上、右上、右下、左下
            auto sortPts = [](cv::Point2f pts[4]) {
                std::vector<cv::Point2f> v(pts, pts + 4);
                std::sort(v.begin(), v.end(), [](cv::Point2f a, cv::Point2f b) {
                    return a.y < b.y;
                });
                // 上面两个，按 x 排序
                if (v[0].x > v[1].x) std::swap(v[0], v[1]);
                // 下面两个，按 x 排序
                if (v[2].x > v[3].x) std::swap(v[2], v[3]);
                return v;
            };

            auto v1 = sortPts(pts1);
            auto v2 = sortPts(pts2);

            // 合并为装甲板的四个角点
            Armor a;
            a.corners = {
                v1[0], v1[1], // 左灯条的上、下
                v2[1], v2[0]  // 右灯条的下、上 (注意顺序对应)
            };
            
            // 计算外接矩形用于画框
            std::vector<cv::Point2f> allPts = {v1[0], v1[1], v1[2], v1[3], v2[0], v2[1], v2[2], v2[3]};
            a.box = cv::boundingRect(allPts);
            a.confidence = 1.0f;
            if (a.box.width < a.box.height) continue;
            armors.push_back(a);
        }
    }

    return armors;
}
