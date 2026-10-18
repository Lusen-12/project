cmake_minimum_required(VERSION 3.10)
project(ArmorDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

file(GLOB SOURCES src/*.cpp)

add_executable(image_eval
    src/image_eval.cpp 
    src/camera.cpp 
    src/pose_solver.cpp 
    src/detector_nn.cpp 
    src/detector_traditional.cpp 
)
target_include_directories(image_eval PRIVATE include /usr/local/include)
target_link_libraries(image_eval ${OpenCV_LIBS} onnxruntime)

add_executable(armor_detector 
    src/main.cpp 
    src/camera.cpp 
    src/pose_solver.cpp 
    src/digit_classifier.cpp 
    src/detector_nn.cpp 
    src/detector_traditional.cpp 
    src/visualizer.cpp
)
target_include_directories(armor_detector PRIVATE include /usr/local/include)
target_link_libraries(armor_detector ${OpenCV_LIBS} onnxruntime)
