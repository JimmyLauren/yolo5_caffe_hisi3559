#include <malloc.h>
#include <string.h>
#include <math.h>
#include "sample_log.h"
#include "hi_runtime_comm.h"
#include "sample_data_utils.h"
#include "sample_memory_ops.h"
#include "sample_cop_param.h"

#define SVP_WK_QUANT_BASE    (0x1000)
#define SVP_WK_COORDI_NUM       (4)
#define SVP_NNIE_MAX_RATIO_ANCHOR_NUM (32) /*NNIE max ratio anchor num*/
#define SAFE_ROUND(val) (double)(((double)(val) > 0)? floor((double)(val)+0.5):ceil((double)(val)-0.5))
static HI_S32 GenBaseAnchor(HI_FLOAT* pf32RatioAnchors, const HI_U32* pu32Ratios, HI_U32 u32NumRatioAnchors, HI_FLOAT* pf32ScaleAnchors,
                            const HI_U32* pu32Scales, HI_U32 u32NumScaleAnchors, const HI_U32* au32BaseAnchor)
{
    /********************* Generate the base anchor ***********************/
    HI_FLOAT f32BaseW = (HI_FLOAT)(au32BaseAnchor[2] - au32BaseAnchor[0] + 1);
    HI_FLOAT f32BaseH = (HI_FLOAT)(au32BaseAnchor[3] - au32BaseAnchor[1] + 1);
    HI_FLOAT f32BaseXCtr = (HI_FLOAT)(au32BaseAnchor[0] + ((f32BaseW - 1) * 0.5));
    HI_FLOAT f32BaseYCtr = (HI_FLOAT)(au32BaseAnchor[1] + ((f32BaseH - 1) * 0.5));
    /*************** Generate Ratio Anchors for the base anchor ***********/
    HI_FLOAT f32Ratios = 0.0f;
    HI_FLOAT f32SizeRatios = 0.0f;
    HI_FLOAT f32Size = f32BaseW * f32BaseH;

    for (HI_U32 i = 0; i < u32NumRatioAnchors; i++)
    {
        f32Ratios = (HI_FLOAT)pu32Ratios[i] / SVP_WK_QUANT_BASE;
        f32SizeRatios = f32Size / f32Ratios;
        f32BaseW = (HI_FLOAT)SAFE_ROUND(sqrt(f32SizeRatios));
        f32BaseH = (HI_FLOAT)SAFE_ROUND(f32BaseW * f32Ratios);
        pf32RatioAnchors[i * SVP_WK_COORDI_NUM + 0] = f32BaseXCtr - (f32BaseW - 1) * 0.5f;
        pf32RatioAnchors[i * SVP_WK_COORDI_NUM + 1] = f32BaseYCtr - (f32BaseH - 1) * 0.5f;
        pf32RatioAnchors[i * SVP_WK_COORDI_NUM + 2] = f32BaseXCtr + (f32BaseW - 1) * 0.5f;
        pf32RatioAnchors[i * SVP_WK_COORDI_NUM + 3] = f32BaseYCtr + (f32BaseH - 1) * 0.5f;
    }

    /********* Generate Scale Anchors for each Ratio Anchor **********/
    /* Generate Scale Anchors for one pixel */
    HI_FLOAT f32Scales = 0.0f;

    for (HI_U32 i = 0; i < u32NumRatioAnchors; i++)
    {
        for (HI_U32 j = 0; j < u32NumScaleAnchors; j++)
        {
            f32BaseW = pf32RatioAnchors[2] - pf32RatioAnchors[0] + 1;
            f32BaseH = pf32RatioAnchors[3] - pf32RatioAnchors[1] + 1;
            f32BaseXCtr = pf32RatioAnchors[0] + (f32BaseW - 1) * 0.5f;
            f32BaseYCtr = pf32RatioAnchors[1] + (f32BaseH - 1) * 0.5f;
            f32Scales = (HI_FLOAT)pu32Scales[j] / SVP_WK_QUANT_BASE;
            pf32ScaleAnchors[0] = f32BaseXCtr - (f32BaseW * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors[1] = f32BaseYCtr - (f32BaseH * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors[2] = f32BaseXCtr + (f32BaseW * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors[3] = f32BaseYCtr + (f32BaseH * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors += SVP_WK_COORDI_NUM;
        }

        pf32RatioAnchors += SVP_WK_COORDI_NUM;
    }

    return HI_SUCCESS;
}


static HI_S32 SetAnchorInPixel(HI_S32* ps32Anchors, const HI_FLOAT* pf32ScaleAnchors, HI_U32 u32ConvHeight, HI_U32 u32ConvWidth, HI_U32 u32NumAnchorPerPixel, HI_U32 u32SpatialScale)
{
    HI_U32 u32anchorCentorX = 0;
    HI_U32 u32anchorCentorY = 0;
    HI_U32 u32ScaleAnchorIndexBase = 0;
    /******************* Copy the anchors to every pixel in the feature map ******************/
    HI_FLOAT f32PixelInterval = SVP_WK_QUANT_BASE / (HI_FLOAT)u32SpatialScale;

    for (HI_U32 h = 0; h < u32ConvHeight; h++)
    {
        for (HI_U32 w = 0; w < u32ConvWidth; w++)
        {
            u32anchorCentorX = (HI_U32)(w * f32PixelInterval);
            u32anchorCentorY = (HI_U32)(h * f32PixelInterval);

            for (HI_U32 n = 0; n < u32NumAnchorPerPixel; n++)
            {
                u32ScaleAnchorIndexBase = n * SVP_WK_COORDI_NUM;
                ps32Anchors[0] = (HI_S32)(u32anchorCentorX + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 0]);
                ps32Anchors[1] = (HI_S32)(u32anchorCentorY + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 1]);
                ps32Anchors[2] = (HI_S32)(u32anchorCentorX + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 2]);
                ps32Anchors[3] = (HI_S32)(u32anchorCentorY + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 3]);
                ps32Anchors += SVP_WK_COORDI_NUM;
            }
        }
    }

    return HI_SUCCESS;
}

HI_S32 createFasterrcnnCopParam(HI_U32 u32ParamNum, HI_RUNTIME_COP_ATTR_S* pCopParam, HI_PROPOSAL_Param_S* pPropsalPram)
{
    HI_FLOAT* pf32RatioAnchors = HI_NULL;
    HI_FLOAT* pf32ScaleAnchors = HI_NULL;
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_MEM_S stAnchorMem;
    HI_U32 au32Ratios[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' ratios*/
    HI_U32 au32Scales[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' scales*/
    HI_U32 u32AnchorMemSize = 0;
    HI_U32 au32BaseAnchor[SVP_WK_COORDI_NUM] = {0, 0, 0, 0};
    SAMPLE_CHK_RETURN((NULL == pPropsalPram || NULL == pCopParam), HI_FAILURE, "param is NULL\n");
    pPropsalPram->stNodePluginParam.u32SrcWidth = 1240;
    pPropsalPram->stNodePluginParam.u32SrcHeight = 375;
    pPropsalPram->stNodePluginParam.enNetType = SVP_NNIE_NET_TYPE_ROI;
    pPropsalPram->stNodePluginParam.u32NumRatioAnchors = 1;
    pPropsalPram->stNodePluginParam.u32NumScaleAnchors = 9;
    pPropsalPram->stNodePluginParam.pfRatio = (HI_FLOAT*)malloc(pPropsalPram->stNodePluginParam.u32NumRatioAnchors * sizeof(HI_FLOAT));
    SAMPLE_CHK_RETURN((NULL == pPropsalPram->stNodePluginParam.pfRatio), HI_FAILURE, "pfRatio malloc error\n");
    pPropsalPram->stNodePluginParam.pfRatio[0] = (HI_FLOAT)2.44;
    pPropsalPram->stNodePluginParam.pfScales = (HI_FLOAT*)malloc(pPropsalPram->stNodePluginParam.u32NumScaleAnchors * sizeof(HI_FLOAT));
    SAMPLE_CHK_GOTO((NULL == pPropsalPram->stNodePluginParam.pfScales), ERROR1, "pfScales malloc error\n");
    pPropsalPram->stNodePluginParam.pfScales[0] = 1.5;
    pPropsalPram->stNodePluginParam.pfScales[1] = (HI_FLOAT)2.1;
    pPropsalPram->stNodePluginParam.pfScales[2] = (HI_FLOAT)2.9;
    pPropsalPram->stNodePluginParam.pfScales[3] = (HI_FLOAT)4.1;
    pPropsalPram->stNodePluginParam.pfScales[4] = (HI_FLOAT)5.8;
    pPropsalPram->stNodePluginParam.pfScales[5] = 8;
    pPropsalPram->stNodePluginParam.pfScales[6] = (HI_FLOAT)11.3;
    pPropsalPram->stNodePluginParam.pfScales[7] = (HI_FLOAT)15.8;
    pPropsalPram->stNodePluginParam.pfScales[8] = (HI_FLOAT)22.1;
    pPropsalPram->stNodePluginParam.u32MaxRoiFrameCnt = 300;
    pPropsalPram->stNodePluginParam.u32MinSize = 16;
    pPropsalPram->stNodePluginParam.fSpatialScale = 0.0625;
    pPropsalPram->stNodePluginParam.fNmsThresh = (HI_FLOAT)0.7;
    pPropsalPram->stNodePluginParam.fFilterThresh = 0;
    pPropsalPram->stNodePluginParam.u32NumBeforeNms = 6000;
    pPropsalPram->stNodePluginParam.fConfThresh = (HI_FLOAT)0.0003;
    pPropsalPram->stNodePluginParam.fValidNmsThresh = (HI_FLOAT)0.3;
    pPropsalPram->stNodePluginParam.u32ClassSize = 2;
    au32BaseAnchor[0] = 0;
    au32BaseAnchor[1] = 0;
    au32BaseAnchor[2] = pPropsalPram->stNodePluginParam.u32MinSize - 1;
    au32BaseAnchor[3] = pPropsalPram->stNodePluginParam.u32MinSize - 1;

    memset(&stAnchorMem, 0x0, sizeof(HI_RUNTIME_MEM_S));
    u32AnchorMemSize = pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       23 * 77 * SVP_WK_COORDI_NUM * sizeof(HI_U32) +
                       pPropsalPram->stNodePluginParam.u32NumRatioAnchors * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT) +
                       pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);

    stAnchorMem.u32Size = u32AnchorMemSize;
    s32Ret = SAMPLE_RUNTIME_HiMemAlloc(&stAnchorMem, HI_TRUE);
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, ERROR2, "alloc Anchor Memory error\n");

    pf32RatioAnchors = (HI_FLOAT*)((HI_UL)(stAnchorMem.u64VirAddr)) + pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       23 * 77 * SVP_WK_COORDI_NUM;
    pf32ScaleAnchors = (HI_FLOAT*)((HI_UL)(stAnchorMem.u64VirAddr)) + pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       23 * 77 * SVP_WK_COORDI_NUM  + pPropsalPram->stNodePluginParam.u32NumRatioAnchors * SVP_WK_COORDI_NUM;

    for (HI_U32 ratioAnchorNum = 0; ratioAnchorNum < pPropsalPram->stNodePluginParam.u32NumRatioAnchors; ++ratioAnchorNum)
    {
        au32Ratios[ratioAnchorNum] = (HI_S32)(pPropsalPram->stNodePluginParam.pfRatio[ratioAnchorNum] * SVP_WK_QUANT_BASE);
    }

    for (HI_U32 scaleAnchorNum = 0; scaleAnchorNum < pPropsalPram->stNodePluginParam.u32NumScaleAnchors; ++scaleAnchorNum)
    {
        au32Scales[scaleAnchorNum] = (HI_S32)(pPropsalPram->stNodePluginParam.pfScales[scaleAnchorNum] * SVP_WK_QUANT_BASE);
    }

    s32Ret = GenBaseAnchor(pf32RatioAnchors, au32Ratios, pPropsalPram->stNodePluginParam.u32NumRatioAnchors,
                           pf32ScaleAnchors, au32Scales, pPropsalPram->stNodePluginParam.u32NumScaleAnchors,
                           au32BaseAnchor);
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, ERROR3, "GenBaseAnchor error\n");

    s32Ret = SetAnchorInPixel((HI_S32*)((HI_UL)(stAnchorMem.u64VirAddr)), pf32ScaleAnchors, 23, 77,
                              pPropsalPram->stNodePluginParam.u32NumScaleAnchors * pPropsalPram->stNodePluginParam.u32NumRatioAnchors,
                              (HI_S32)(pPropsalPram->stNodePluginParam.fSpatialScale * SVP_WK_QUANT_BASE));
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, ERROR3, "SetAnchorInPixel error\n");

    memcpy(&(pPropsalPram->stBachorMemInfo), &stAnchorMem, sizeof(HI_RUNTIME_MEM_S));
    pCopParam[0].pConstParam = (HI_VOID*)pPropsalPram;
    pCopParam[0].u32ConstParamSize = sizeof(HI_PROPOSAL_Param_S);
    memset(pCopParam[0].acCopName, 0, MAX_NAME_LEN + 1);
    strncpy(pCopParam[0].acCopName, "proposal", MAX_NAME_LEN);

    return HI_SUCCESS;
ERROR3:
    SAMPLE_FreeMem(&stAnchorMem);
ERROR2:
    SAMPLE_FREE(pPropsalPram->stNodePluginParam.pfScales);
ERROR1:
    SAMPLE_FREE(pPropsalPram->stNodePluginParam.pfRatio);
    return HI_FAILURE;
}

HI_S32 createRFCNCopParam(HI_U32 u32ParamNum, HI_RUNTIME_COP_ATTR_S* pCopParam, HI_PROPOSAL_Param_S* pPropsalPram)
{

    HI_FLOAT* pf32RatioAnchors = HI_NULL;
    HI_FLOAT* pf32ScaleAnchors = HI_NULL;
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 au32Ratios[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' ratios*/
    HI_U32 au32Scales[SVP_NNIE_MAX_RATIO_ANCHOR_NUM];  /*anchors' scales*/
    HI_U32 u32AnchorMemSize = 0;
    HI_RUNTIME_MEM_S stAnchorMem;
    HI_U32 au32BaseAnchor[SVP_WK_COORDI_NUM] = { 0, 0, 0, 0};

    pPropsalPram->stNodePluginParam.u32SrcWidth = 800;
    pPropsalPram->stNodePluginParam.u32SrcHeight = 600;
    pPropsalPram->stNodePluginParam.enNetType = SVP_NNIE_NET_TYPE_ROI;
    pPropsalPram->stNodePluginParam.u32NumRatioAnchors = 3;
    pPropsalPram->stNodePluginParam.u32NumScaleAnchors = 3;
    pPropsalPram->stNodePluginParam.pfRatio = (HI_FLOAT*)malloc(pPropsalPram->stNodePluginParam.u32NumRatioAnchors * sizeof(HI_FLOAT));
    SAMPLE_CHK_RETURN((NULL == pPropsalPram->stNodePluginParam.pfRatio), HI_FAILURE, "pfRatio malloc error\n");
    pPropsalPram->stNodePluginParam.pfRatio[0] = 0.5;
    pPropsalPram->stNodePluginParam.pfRatio[1] = 1;
    pPropsalPram->stNodePluginParam.pfRatio[2] = 2;
    pPropsalPram->stNodePluginParam.pfScales = (HI_FLOAT*)malloc(pPropsalPram->stNodePluginParam.u32NumScaleAnchors * sizeof(HI_FLOAT));
    SAMPLE_CHK_GOTO((NULL == pPropsalPram->stNodePluginParam.pfScales), ERROR1, "pfScales malloc error\n");
    pPropsalPram->stNodePluginParam.pfScales[0] = 8;
    pPropsalPram->stNodePluginParam.pfScales[1] = 16;
    pPropsalPram->stNodePluginParam.pfScales[2] = 32;
    pPropsalPram->stNodePluginParam.u32MaxRoiFrameCnt = 300;
    pPropsalPram->stNodePluginParam.u32MinSize = 16;
    pPropsalPram->stNodePluginParam.fSpatialScale = 0.0625;
    pPropsalPram->stNodePluginParam.fNmsThresh = 0.7;
    pPropsalPram->stNodePluginParam.fFilterThresh = 0;
    pPropsalPram->stNodePluginParam.u32NumBeforeNms = 6000;
    pPropsalPram->stNodePluginParam.fConfThresh = 0.3;
    pPropsalPram->stNodePluginParam.fValidNmsThresh = 0.3;
    pPropsalPram->stNodePluginParam.u32ClassSize = 21;
    au32BaseAnchor[0] = 0;
    au32BaseAnchor[1] = 0;
    au32BaseAnchor[2] = pPropsalPram->stNodePluginParam.u32MinSize - 1;
    au32BaseAnchor[3] = pPropsalPram->stNodePluginParam.u32MinSize - 1;

    memset(&stAnchorMem, 0x0, sizeof(HI_RUNTIME_MEM_S));
    u32AnchorMemSize = pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       38 * 50 * SVP_WK_COORDI_NUM * sizeof(HI_U32) +
                       pPropsalPram->stNodePluginParam.u32NumRatioAnchors * SVP_WK_COORDI_NUM * sizeof(HI_FLOAT) +
                       pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       SVP_WK_COORDI_NUM * sizeof(HI_FLOAT);

    stAnchorMem.u32Size = u32AnchorMemSize;
    s32Ret = SAMPLE_RUNTIME_HiMemAlloc(&stAnchorMem, HI_TRUE);
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, ERROR2, "alloc Anchor Memory error\n");

    pf32RatioAnchors = (HI_FLOAT*)((HI_UL)(stAnchorMem.u64VirAddr)) + pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       38 * 50 * SVP_WK_COORDI_NUM;
    pf32ScaleAnchors = (HI_FLOAT*)((HI_UL)(stAnchorMem.u64VirAddr)) + pPropsalPram->stNodePluginParam.u32NumRatioAnchors * pPropsalPram->stNodePluginParam.u32NumScaleAnchors *
                       38 * 50 * SVP_WK_COORDI_NUM  + pPropsalPram->stNodePluginParam.u32NumRatioAnchors * SVP_WK_COORDI_NUM;

    for (HI_U32 ratioAnchorNum = 0; ratioAnchorNum < pPropsalPram->stNodePluginParam.u32NumRatioAnchors; ++ratioAnchorNum)
    {
        au32Ratios[ratioAnchorNum] = (HI_S32)(pPropsalPram->stNodePluginParam.pfRatio[ratioAnchorNum] * SVP_WK_QUANT_BASE);
    }

    for (HI_U32 scaleAnchorNum = 0; scaleAnchorNum < pPropsalPram->stNodePluginParam.u32NumScaleAnchors; ++scaleAnchorNum)
    {
        au32Scales[scaleAnchorNum] = (HI_S32)(pPropsalPram->stNodePluginParam.pfScales[scaleAnchorNum] * SVP_WK_QUANT_BASE);
    }

    s32Ret = GenBaseAnchor(pf32RatioAnchors, au32Ratios, pPropsalPram->stNodePluginParam.u32NumRatioAnchors,
                           pf32ScaleAnchors, au32Scales, pPropsalPram->stNodePluginParam.u32NumScaleAnchors,
                           au32BaseAnchor);
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, ERROR3, "GenBaseAnchor error\n");

    s32Ret = SetAnchorInPixel((HI_S32*)((HI_UL)(stAnchorMem.u64VirAddr)), pf32ScaleAnchors, 38, 50,
                              pPropsalPram->stNodePluginParam.u32NumScaleAnchors * pPropsalPram->stNodePluginParam.u32NumRatioAnchors,
                              (HI_S32)(pPropsalPram->stNodePluginParam.fSpatialScale * SVP_WK_QUANT_BASE));
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, ERROR3, "SetAnchorInPixel error\n");

    memcpy(&(pPropsalPram->stBachorMemInfo), &stAnchorMem, sizeof(HI_RUNTIME_MEM_S));
    pCopParam[0].pConstParam = (HI_VOID*)pPropsalPram;
    pCopParam[0].u32ConstParamSize = sizeof(HI_PROPOSAL_Param_S);
    memset(pCopParam[0].acCopName, 0, MAX_NAME_LEN + 1);
    strncpy(pCopParam[0].acCopName, "proposal", MAX_NAME_LEN);

    return HI_SUCCESS;
ERROR3:
    SAMPLE_FreeMem(&stAnchorMem);
ERROR2:
    SAMPLE_FREE(pPropsalPram->stNodePluginParam.pfScales);
ERROR1:
    SAMPLE_FREE(pPropsalPram->stNodePluginParam.pfRatio);
    return HI_FAILURE;
}

HI_VOID releaseRfcnAndFrcnnCopParam(HI_U32 u32ParamNum, HI_PROPOSAL_Param_S* pCopParam)
{
    SAMPLE_FREE(pCopParam[0].stNodePluginParam.pfScales);
    SAMPLE_FREE(pCopParam[0].stNodePluginParam.pfRatio);
    HI_RUNTIME_MEM_S stMem;
    stMem.u64PhyAddr = pCopParam[0].stBachorMemInfo.u64PhyAddr;
    stMem.u64VirAddr = pCopParam[0].stBachorMemInfo.u64VirAddr;
    SAMPLE_FreeMem(&stMem);
}
