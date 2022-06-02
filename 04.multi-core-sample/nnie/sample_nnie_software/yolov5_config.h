//
// Created by jimmy on 8/27/21.
//

#ifndef FD_CONFIG
#define FD_CONFIG

#include <stdio.h>
//#include"list.h"
#include<math.h>
#include "sample_comm_nnie.h"

#define QUANT_BASE 4096.0f

#define yolo_layer_num 3 // yolo layer 层数

#ifndef YOLO_MIN
#define YOLO_MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef YOLO_MAX
#define YOLO_MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#define IMAGE_W 416.0f // 输入图片大小
#define IMAGE_H 416.0f

typedef struct anchor_w_h
{
    float anchor_w;
    float anchor_h;
}anchor_w_h;

typedef struct yolo_result
{
    float left_up_x;
    float left_up_y;
    float right_down_x;
    float right_down_y;
    int class_index;
    float score;
    struct yolo_result* next;
}yolo_result;


anchor_w_h  anchor_grids[3][3] = {{{3.0f, 3.0f}, {5.0f, 5.0f}, {9.0f, 8.0f}}, // small yolo layer 层 anchor
                                  {{7.0f, 13.0f}, {13.0f, 12.0f}, {20.0f, 16.0f}}, // middle yolo layer 层 anchor
                                  {{26.0f, 27.0f}, {48.0f, 42.0f}, {98.0f, 98.0f}}}; // large yolo layer 层 anchor


float strides[3] = {8.0f, 16.0f, 32.0f}; // 每个 yolo 层，grid 大小，与上面顺序对应
int map_size[3] = {52, 26, 13}; // 每个 yolo 层，feature map size 大小，与上面顺序对应

#endif // FD_CONFIG

