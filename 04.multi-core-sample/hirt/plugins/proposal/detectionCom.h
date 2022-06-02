#ifndef _DETECTION_COM_H_
#define _DETECTION_COM_H_

#include "back/hi_type.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#define _mkdir(a) mkdir((a),0755)
#endif

#define DETECION_DBG (0)

#define SVP_WK_PROPOSAL_WIDTH   (6)
#define SVP_WK_COORDI_NUM       (4)
#define SVP_WK_SCORE_NUM        (2)
#define MAX_STACK_DEPTH         (50000)

#define SVP_NNIE_MAX_REPORT_NODE_CNT  (16) /*NNIE max report num*/
#define SVP_NNIE_MAX_RATIO_ANCHOR_NUM (32) /*NNIE max ratio anchor num*/

#ifndef SVP_WK_QUANT_BASE
#define SVP_WK_QUANT_BASE (0x1000)
#endif

#ifndef ALIGN32
#define ALIGN32(addr) ((((addr) + 32 - 1)/32)*32)
#endif

#ifndef ALIGN16
#define ALIGN16(addr) ((((addr) + 16 - 1)/16)*16)
#endif

#ifndef SVP_MAX
#define SVP_MAX(a,b) (((a)>(b))?(a):(b))
#endif

#ifndef SVP_MIN
#define SVP_MIN(a,b) (((a)<(b))?(a):(b))
#endif

#define SAFE_ROUND(val) (double)(((double)(val) > 0)? floor((double)(val)+0.5):ceil((double)(val)-0.5))

#define SVP_FALSE_CHECK(cond, ec) \
    do{\
        if (!(cond)) {\
            printf("%s %d CHECK error! cond = %d, do ret = %d\n", __FILE__, __LINE__, cond, ec);\
            return ec;\
        }\
    }while(0)

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

enum RPN_SUPRESS_FLAG
{
    RPN_SUPPRESS_FALSE = 0,
    RPN_SUPPRESS_TRUE = 1,

    RPN_SUPPRESS_BUTT
};

typedef struct hiNNIE_STACK
{
    HI_S32 s32Min;
    HI_S32 s32Max;
} NNIE_STACK_S;

typedef struct hiSVP_NNIE_SRC_NODE_S
{
    HI_U32 u32Height;
    HI_U32 u32Width;
    HI_U32 u32Chn;
    HI_U8  u8Format;
    HI_U8  u8Reserved;
    HI_U16 u16LayerId;
} SVP_NNIE_SRC_NODE_S;

typedef struct hiSVP_NNIE_ROI_NODE_S
{
    HI_U32 u32SrcPoolHeight;
    HI_U32 u32SrcPoolWidth;
    float  f32Scale;
    HI_U8  u8UsePingPong;
    HI_U8  u8IsHighPrecision;
    HI_U8  u8RoiPoolType;
    HI_U8  RSV1;

    HI_U32 u32DstPoolHeight;
    HI_U32 u32DstPoolWidth;
    HI_U32 u32Channel;
    HI_U32 u32MaxRoiNum;

    HI_U32 u32BlockNum;
    HI_U32 u32BlockHeight;
    HI_U32 u32MaxROIInfoSize;
    HI_U32 RSV2;
} SVP_NNIE_ROI_NODE_S;


/*Network type*/
typedef enum hiSVP_NNIE_NET_TYPE_E
{
    SVP_NNIE_NET_TYPE_CNN = 0x0, /* Non-ROI input cnn net */
    SVP_NNIE_NET_TYPE_ROI = 0x1, /* With ROI input cnn net*/
    SVP_NNIE_NET_TYPE_RECURRENT = 0x2, /* RNN or LSTM net */

    SVP_NNIE_NET_TYPE_BUTT
} SVP_NNIE_NET_TYPE_E;

typedef struct hiNNIE_REPORT_NODE_INFO_S
{
    HI_U32 u32ConvWidth;     /* width */
    HI_U32 u32ConvHeight;    /* height */
    HI_U32 u32ConvMapNum;    /* map num */
    HI_U32 u32ConvStride;    /* stride */
} NNIE_REPORT_NODE_INFO_S;


typedef struct hiNNIE_MODEL_INFO_S
{
    /*MPI layer input*/
    HI_U32 u32MemPoolSize;              /*memory pool size*/
    SVP_NNIE_NET_TYPE_E  enNetType;     /*net type*/

    HI_U32 u32SrcWidth;                 /*input pic width*/
    HI_U32 u32SrcHeight;                /*input pic height*/
    HI_U32 u32SrcStride;                /*input pic stride*/

    NNIE_REPORT_NODE_INFO_S  astReportNodeInfo[SVP_NNIE_MAX_REPORT_NODE_CNT];  /*report node info*/
    HI_U32 u32ReportNodeNum;                           /*report node number*/

    HI_U32 u32MinSize;                                 /*min anchor size*/
    HI_U32 u32SpatialScale;                            /*spatial scale*/
    HI_U32 au32Ratios[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' ratios*/
    HI_U32 u32NumRatioAnchors;                         /*num of ratio anchors*/
    HI_U32 au32Scales[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' scales*/
    HI_U32 u32NumScaleAnchors;                         /*num of scale anchors*/

    HI_U32 u32RoiWidth;                                /*rcnn roi width*/
    HI_U32 u32RoiHeight;                               /*rcnn roi height*/
    HI_U32 u32RoiMapNum;                               /*rcnn roi map num*/
    HI_U32 u32RoiStride;                               /*rcnn roi stride*/
    HI_U32 u32MaxRoiFrameCnt;                          /*max roi frame cnt*/

    HI_U32 u32DnnChannelNum;                           /*dnn input channel num, current version rsv*/
    HI_U32 u32ChannelNum;
    HI_U8  u8RunMode;

    // support pooling report
    HI_U32 u32ReportMode;                              /*final report mode: 0-fc report, 1-conv or pooling report*/
    HI_U32 u32ClassSize;                               /*class category*/
    HI_U32 u32ClassStride;                             /*class stride*/
} NNIE_MODEL_INFO_S;


//typedef std::pair<std::string, std::string> SVP_SAMPLE_FILE_NAME_PAIR;

/*********************************************************
Function: QuickExp
Description: Do QuickExp...
*********************************************************/
HI_FLOAT QuickExp(HI_U32 u32X);

/*********************************************************
Function: SoftMax
Description: Do softmax on a vector of length s32ArraySize
*********************************************************/
HI_S32 SoftMax(HI_FLOAT* af32Src, HI_S32 s32ArraySize);
/*deal with num*/
HI_S32 SoftMax_N(HI_FLOAT* af32Src, HI_S32 s32ArraySize, HI_U32 u32Num);

/**************************************************
Function: Argswap
Description: used in NonRecursiveQuickSort
***************************************************/
HI_S32 Argswap(HI_S32* ps32Src1, HI_S32* ps32Src2);

/**************************************************
Function: NonRecursiveArgQuickSort
Description: sort with NonRecursiveArgQuickSort
***************************************************/
HI_S32 NonRecursiveArgQuickSort(HI_S32* aResultArray, HI_S32 s32Low, HI_S32 s32High, NNIE_STACK_S* pstStack, HI_U32 u32MaxNum);

/**************************************************
Function: NonMaxSuppression
Description: proposal NMS with u32NmsThresh
***************************************************/
HI_S32 NonMaxSuppression(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh, HI_U32 u32MaxRoiNum);

/**************************************************
Function: FilterLowScoreBbox
Description: remove low conf score proposal bbox
***************************************************/
HI_S32 FilterLowScoreBbox(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh,
                          HI_U32 u32FilterThresh, HI_U32* u32NumAfterFilter);

/**************************************************
Function: generate Base Anchors
Description: generate Base Anchors by give miniSize, ratios, and scales
***************************************************/
HI_S32 GenBaseAnchor(
    HI_FLOAT* pf32RatioAnchors, const HI_U32* pu32Ratios, HI_U32 u32NumRatioAnchors,
    HI_FLOAT* pf32ScaleAnchors, const HI_U32* pu32Scales, HI_U32 u32NumScaleAnchors,
    const HI_U32* au32BaseAnchor);

/**************************************************
Function: SetAnchorInPixel
Description: set base anchor to origin pic point based on pf32ScaleAnchors
***************************************************/
HI_S32 SetAnchorInPixel(
    HI_S32* ps32Anchors,
    const HI_FLOAT* pf32ScaleAnchors,
    HI_U32 u32ConvHeight,
    HI_U32 u32ConvWidth,
    HI_U32 u32NumAnchorPerPixel,
    HI_U32 u32SpatialScale);

/**************************************************
Function: BBox Transform
Description: parameters from Conv3 to adjust the coordinates of anchor
***************************************************/
HI_S32 BboxTransform(
    HI_S32* ps32Proposals,
    HI_S32* ps32Anchors,
    HI_S32* ps32BboxDelta,
    HI_FLOAT* pf32Scores);

/*deal with num*/
HI_S32 BboxTransform_N(
    HI_S32* ps32Proposals,
    HI_S32* ps32Anchors,
    HI_S32* ps32BboxDelta,
    HI_FLOAT* pf32Scores,
    HI_U32 u32NumAnchors);

HI_S32 BboxTransform_FLOAT(
    HI_FLOAT* pf32Proposals,
    HI_FLOAT* pf32Anchors,
    HI_FLOAT* pf32BboxDelta,
    HI_FLOAT* pf32Scores);

/**************************************************
Function: BboxClip
Description: clip proposal bbox out of origin image range
***************************************************/

HI_S32 BboxClip(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH);

/*deal with num*/
HI_S32 BboxClip_N(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH, HI_U32 u32Num);

/* single size clip */
HI_S32 SizeClip(HI_S32 s32inputSize, HI_S32 s32sizeMin, HI_S32 s32sizeMax);

/**************************************************
Function: BboxSmallSizeFilter
Description: remove the bboxes which are too small
***************************************************/
HI_S32 BboxSmallSizeFilter(HI_S32* ps32Proposals, HI_U32 u32minW, HI_U32 u32minH);
HI_S32 BboxSmallSizeFilter_N(HI_S32* ps32Proposals, HI_U32 u32minW, HI_U32 u32minH, HI_U32 u32NumAnchors);

/**************************************************
Function: dumpProposal
Description: dumpProposal info when DETECION_DBG
***************************************************/
HI_S32 dumpProposal(HI_S32* ps32Proposals, const HI_CHAR* filename, HI_U32 u32NumAnchors);

/**************************************************
Function: getRPNresult
Description: rite the final result to output
***************************************************/
HI_S32 getRPNresult(HI_S32* ps32ProposalResult, HI_U32* pu32NumRois, HI_U32 u32MaxRois,
                    const HI_S32* ps32Proposals, HI_U32 u32NumAfterFilter);

/**************************************************
Function: BreakLine
Description:
***************************************************/
void PrintBreakLine(HI_BOOL flag);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
