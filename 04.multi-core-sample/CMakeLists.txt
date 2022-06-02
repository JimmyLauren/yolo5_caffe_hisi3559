cmake_minimum_required(VERSION 3.15)
project(multi_core_sample)

set(CMAKE_CXX_STANDARD 14)



include_directories(
        ./nnie/
        ./nnie/sample
        ./nnie/sample_nnie_software
        ./common
        ./ive/sample
        ./ive/
        ./opencv/include
        ./opencv/include/opencv
        ./opencv/include/opencv2
        ./thirdparty/include/hisi
)



add_executable(multi_core_sample ./nnie/sample_nnie_main.c nnie/sample_nnie_software/yolov5_config.h ive/sample/sample_ive_add.c)