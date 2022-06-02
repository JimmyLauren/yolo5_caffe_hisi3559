#include <stdlib.h>
#include <string.h>
#include "hi_plugin.h"
#include "hi_runtime_comm.h"
#include "sample_log.h"
#include "sample_memory_ops.h"
#include "stdio.h"
#include "math.h"
#include "detectionCom.h"

#define SVP_WK_QUANT_BASE (0x1000)
#define SVP_WK_COORDI_NUM       (4)
#define SVP_WK_PROPOSAL_WIDTH   (6)
#define SVP_WK_SCORE_NUM        (2)
#define MAX_STACK_DEPTH         (50000)
#define SVP_NNIE_MAX_RATIO_ANCHOR_NUM (32) /*NNIE max ratio anchor num*/
#define SVP_NNIE_MAX_REPORT_NODE_CNT  (16) /*NNIE max report num*/
#define RPN_NODE_NUM  (3)

typedef struct hiNodePlugin_Param_S
{
    HI_U32 u32SrcWidth;
    HI_U32 u32SrcHeight;
    SVP_NNIE_NET_TYPE_E enNetType;
    HI_U32 u32NumRatioAnchors;
    HI_U32 u32NumScaleAnchors;
    HI_FLOAT* pfRatio;
    HI_FLOAT* pfScales;
    HI_U32 u32MaxRoiFrameCnt;
    HI_U32 u32MinSize;
    HI_FLOAT fSpatialScale;
    HI_FLOAT fNmsThresh;
    HI_FLOAT fFilterThresh;
    HI_U32 u32NumBeforeNms;
    HI_FLOAT fConfThresh;
    HI_FLOAT fValidNmsThresh;
    HI_U32 u32ClassSize;
} HI_NODEPlugin_Param_S;

typedef struct hiProposal_param_s
{
    HI_RUNTIME_MEM_S stBachorMemInfo;
    HI_NODEPlugin_Param_S stNodePluginParam;
} HI_PROPOSAL_Param_S;

typedef struct _PROPOSAL_PARA_S
{
    //---------- parameters for PriorBox ---------
    NNIE_MODEL_INFO_S model_info;
    HI_U32 u32NumBeforeNms;
    HI_U32 u32NmsThresh;
    HI_U32 u32ValidNmsThresh;
    HI_U32 u32FilterThresh;
    HI_U32 u32ConfThresh;
    HI_RUNTIME_MEM_S stBachorMemInfo;
} PROPOSAL_PARA_S;

static HI_VOID getParam(const HI_PROPOSAL_Param_S* pPluginParam, PROPOSAL_PARA_S* param)
{
    if (HI_NULL == pPluginParam) return;
    param->model_info.u32SrcWidth = pPluginParam->stNodePluginParam.u32SrcWidth;
    param->model_info.u32SrcHeight = pPluginParam->stNodePluginParam.u32SrcHeight;
    param->model_info.enNetType = pPluginParam->stNodePluginParam.enNetType;
    param->model_info.u32NumRatioAnchors = pPluginParam->stNodePluginParam.u32NumRatioAnchors;
    param->model_info.u32NumScaleAnchors = pPluginParam->stNodePluginParam.u32NumScaleAnchors;
    for (HI_U32 ratioAnchorNum = 0; ratioAnchorNum < pPluginParam->stNodePluginParam.u32NumRatioAnchors; ++ratioAnchorNum) {
        param->model_info.au32Ratios[ratioAnchorNum] = (HI_S32)(pPluginParam->stNodePluginParam.pfRatio[ratioAnchorNum] * SVP_WK_QUANT_BASE);
    }
    for (HI_U32 scaleAnchorNum = 0; scaleAnchorNum < pPluginParam->stNodePluginParam.u32NumScaleAnchors; ++scaleAnchorNum) {
        param->model_info.au32Scales[scaleAnchorNum] = (HI_S32)(pPluginParam->stNodePluginParam.pfScales[scaleAnchorNum] * SVP_WK_QUANT_BASE);
    }

    param->model_info.u32ClassSize = pPluginParam->stNodePluginParam.u32ClassSize;
    param->model_info.u32MaxRoiFrameCnt = pPluginParam->stNodePluginParam.u32MaxRoiFrameCnt;
    param->model_info.u32MinSize = pPluginParam->stNodePluginParam.u32MinSize;
    param->model_info.u32SpatialScale = (HI_S32)(pPluginParam->stNodePluginParam.fSpatialScale * SVP_WK_QUANT_BASE);
    param->u32NmsThresh = (HI_S32)(pPluginParam->stNodePluginParam.fNmsThresh * SVP_WK_QUANT_BASE);
    param->u32FilterThresh = (HI_U32)(pPluginParam->stNodePluginParam.fFilterThresh * SVP_WK_QUANT_BASE);
    param->u32NumBeforeNms = pPluginParam->stNodePluginParam.u32NumBeforeNms;
    param->u32ConfThresh = (HI_U32)(pPluginParam->stNodePluginParam.fConfThresh * SVP_WK_QUANT_BASE);
    param->u32ValidNmsThresh = (HI_U32)(pPluginParam->stNodePluginParam.fValidNmsThresh * SVP_WK_QUANT_BASE);
    memcpy(&(param->stBachorMemInfo), &(pPluginParam->stBachorMemInfo), sizeof(HI_RUNTIME_MEM_S));
}

static HI_U32 GetRFCNAssistMemSize(PROPOSAL_PARA_S* para)
{
    HI_U32 u32NumAnchors = (para->model_info.u32NumRatioAnchors) *
        (para->model_info.u32NumScaleAnchors)*
        (para->model_info.astReportNodeInfo[0].u32ConvHeight) *
        (para->model_info.astReportNodeInfo[0].u32ConvWidth);

    HI_U32 u32AnchorSize = u32NumAnchors* SVP_WK_COORDI_NUM * sizeof(HI_U32);
    HI_U32 u32BboxDeltaSize = u32AnchorSize;
    HI_U32 u32ProposalSize = u32NumAnchors * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_U32);
    HI_U32 u32ScoresSize = u32NumAnchors * SVP_WK_SCORE_NUM * sizeof(HI_FLOAT);
    HI_U32 u32StackSize = MAX_STACK_DEPTH * sizeof(NNIE_STACK_S);
    HI_U32 u32TotalSize = u32BboxDeltaSize + u32ProposalSize + u32ScoresSize + u32StackSize;


    return u32TotalSize;
}

static HI_S32 HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref(
    HI_S32 *ps32Src[],
    HI_S32 s32SrcNum,
    PROPOSAL_PARA_S *pProposalParam,
    HI_U32 *pu32MemPool,
    HI_S32 *ps32ProposalResult,
    HI_U32 *pu32NumRois)
{
    /******************** define parameters ****************/
    HI_U32 u32Size = 0;
    HI_S32* ps32Anchors = NULL;
    HI_S32* ps32BboxDelta = NULL;
    HI_S32* ps32Proposals = NULL;
    HI_U32 u32OriImHeight = 0;
    HI_U32 u32OriImWidth = 0;
    HI_U32* pu32Ptr = NULL;

    HI_U32 u32MaxRois = 0;
    HI_U32 u32NumAfterFilter = 0;
    HI_U32 u32NumAnchors = 0;
    HI_FLOAT* pf32Ptr = NULL;
    HI_FLOAT* pf32Scores = NULL;

    HI_U32 u32SrcBboxIndex = 0;
    HI_U32 u32SrcFgProbIndex = 0;
    HI_U32 u32SrcBgProbIndex = 0;

    HI_U32 u32SrcProbBias = 0;
    HI_U32 u32DesBox = 0;
    HI_U32 u32BgBlobSize = 0;
    HI_U32 u32AnchorsPerPixel = 0;
    HI_U32 u32MapSize = 0;
    HI_U32 u32LineSize = 0;

    HI_U32 u32DesBboxDeltaIndex = 0;
    HI_U32 u32DesScoreIndex = 0;

    NNIE_STACK_S* pstStack = NULL;
    HI_S32 s32Ret = HI_FAILURE;

    NNIE_REPORT_NODE_INFO_S astReportNode[RPN_NODE_NUM];
    memset(astReportNode, 0, sizeof(NNIE_REPORT_NODE_INFO_S) * RPN_NODE_NUM);

    /******************** Get parameters from Model and Config ***********************/
    u32OriImHeight = pProposalParam->model_info.u32SrcHeight;
    u32OriImWidth = pProposalParam->model_info.u32SrcWidth;

    if (SVP_NNIE_NET_TYPE_ROI != pProposalParam->model_info.enNetType)
    {
        for (HI_U32 i = 0; i < RPN_NODE_NUM; i++)
        {
            astReportNode[i].u32ConvHeight = pProposalParam->model_info.astReportNodeInfo[i].u32ConvHeight;
            astReportNode[i].u32ConvWidth = pProposalParam->model_info.astReportNodeInfo[i].u32ConvWidth;
            astReportNode[i].u32ConvMapNum = pProposalParam->model_info.astReportNodeInfo[i].u32ConvMapNum;
            astReportNode[i].u32ConvStride = pProposalParam->model_info.astReportNodeInfo[i].u32ConvStride;
        }
    }
    else
    {
        for (HI_U32 i = 0; i < RPN_NODE_NUM - 1; i++)
        {
            astReportNode[i + 1].u32ConvHeight = pProposalParam->model_info.astReportNodeInfo[i].u32ConvHeight;
            astReportNode[i + 1].u32ConvWidth = pProposalParam->model_info.astReportNodeInfo[i].u32ConvWidth;
            astReportNode[i + 1].u32ConvMapNum = pProposalParam->model_info.astReportNodeInfo[i].u32ConvMapNum;
            astReportNode[i + 1].u32ConvStride = pProposalParam->model_info.astReportNodeInfo[i].u32ConvStride;
        }
    }

    u32MaxRois = pProposalParam->model_info.u32MaxRoiFrameCnt;


    /*********************************** Faster RCNN *********************************************/
    /********* calculate the start pointer of each part in MemPool *********/
    /* base RatioAnchors and ScaleAnchors */
    u32NumAnchors = (pProposalParam->model_info.u32NumRatioAnchors) *
        (pProposalParam->model_info.u32NumScaleAnchors) *
        (astReportNode[1].u32ConvHeight * astReportNode[1].u32ConvWidth);
    u32Size = SVP_WK_COORDI_NUM * u32NumAnchors;

    ps32Anchors = (HI_S32*)((HI_UL)(pProposalParam->stBachorMemInfo.u64VirAddr));
    pu32Ptr = (HI_U32*)pu32MemPool;
    /* BboxDelta */
    ps32BboxDelta = (HI_S32*)pu32Ptr;
    pu32Ptr += u32Size;

    /* Proposal info */
    ps32Proposals = (HI_S32*)pu32Ptr;
    u32Size = SVP_WK_PROPOSAL_WIDTH * u32NumAnchors;
    pu32Ptr += u32Size;



    pf32Ptr = (HI_FLOAT*)pu32Ptr;
    /* Proposal scores */
    pf32Scores = pf32Ptr;
    u32Size = u32NumAnchors * SVP_WK_SCORE_NUM;
    pf32Ptr += u32Size;

    /* quick sort Stack */
    pstStack = (NNIE_STACK_S*)pf32Ptr;


    /********** do transpose, convert the blob from (M,C,H,W) to (M,H,W,C) **********/
    u32MapSize = (astReportNode[2].u32ConvHeight) * (astReportNode[2].u32ConvStride / sizeof(HI_U32));
    u32AnchorsPerPixel = pProposalParam->model_info.u32NumRatioAnchors * pProposalParam->model_info.u32NumScaleAnchors;
    u32BgBlobSize = u32AnchorsPerPixel * u32MapSize;
    u32LineSize = (astReportNode[2].u32ConvStride) / sizeof(HI_U32);

    u32SrcProbBias = (astReportNode[0].u32ConvMapNum) *
        (astReportNode[0].u32ConvHeight) *
        (astReportNode[1].u32ConvStride / sizeof(HI_U32)); /* skip the 1st report node */

    for (HI_U32 c = 0; c < astReportNode[2].u32ConvMapNum; c++)
    {
        for (HI_U32 h = 0; h < astReportNode[2].u32ConvHeight; h++)
        {
            for (HI_U32 w = 0; w < astReportNode[2].u32ConvWidth; w++)
            {
                u32SrcBgProbIndex = u32SrcProbBias + (c / SVP_WK_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcBboxIndex = c * u32MapSize + h * u32LineSize + w;
                u32SrcBgProbIndex = (c / SVP_WK_COORDI_NUM) * u32MapSize + h * u32LineSize + w;
                u32SrcFgProbIndex = u32BgBlobSize + u32SrcBgProbIndex;

                u32DesBox = (u32AnchorsPerPixel) * (h * astReportNode[2].u32ConvWidth + w) + (c / SVP_WK_COORDI_NUM);

                u32DesBboxDeltaIndex = SVP_WK_COORDI_NUM * u32DesBox + (c % SVP_WK_COORDI_NUM);
                ps32BboxDelta[u32DesBboxDeltaIndex] = ps32Src[1][u32SrcBboxIndex];

                u32DesScoreIndex = SVP_WK_SCORE_NUM * u32DesBox;
                pf32Scores[u32DesScoreIndex + 0] = (HI_FLOAT)ps32Src[0][u32SrcBgProbIndex] / SVP_WK_QUANT_BASE;
                pf32Scores[u32DesScoreIndex + 1] = (HI_FLOAT)ps32Src[0][u32SrcFgProbIndex] / SVP_WK_QUANT_BASE;
            }
        }
    }

    /************************* do softmax ****************************/
    s32Ret = SoftMax_N(pf32Scores, SVP_WK_SCORE_NUM, u32NumAnchors);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "softmax error, error code is %d", s32Ret);

    /************************* BBox Transform *****************************/
    s32Ret = BboxTransform_N(ps32Proposals, ps32Anchors, ps32BboxDelta, pf32Scores, u32NumAnchors);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "BboxTransform_N error, error code is %d", s32Ret);

    /************************ clip bbox *****************************/
    s32Ret = BboxClip_N(ps32Proposals, u32OriImWidth, u32OriImHeight, u32NumAnchors);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "BboxClip_N error, error code is %d", s32Ret);

    /************ remove the bboxes which are too small ***********/
    s32Ret = BboxSmallSizeFilter_N(ps32Proposals, pProposalParam->model_info.u32MinSize, pProposalParam->model_info.u32MinSize, u32NumAnchors);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "BboxSmallSizeFilter_N error, error code is %d", s32Ret);

    /********** remove low score bboxes ************/
    s32Ret = FilterLowScoreBbox(ps32Proposals, u32NumAnchors, pProposalParam->u32NmsThresh, pProposalParam->u32FilterThresh, &u32NumAfterFilter);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "FilterLowScoreBbox error, error code is %d", s32Ret);

    /********** sort ***********/
    s32Ret = NonRecursiveArgQuickSort(ps32Proposals, 0, (HI_S32)u32NumAfterFilter - 1, pstStack, pProposalParam->u32NumBeforeNms);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "NonRecursiveArgQuickSort error, error code is %d", s32Ret);

    u32NumAfterFilter = SVP_MIN(u32NumAfterFilter, pProposalParam->u32NumBeforeNms);

    /* do nms to remove highly overlapped bbox */
    s32Ret = NonMaxSuppression(ps32Proposals, u32NumAfterFilter, pProposalParam->u32NmsThresh, pProposalParam->model_info.u32MaxRoiFrameCnt);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "NonMaxSuppression error, error code is %d", s32Ret);

    /************** write the final result to output ***************/
    s32Ret = getRPNresult(ps32ProposalResult, pu32NumRois, u32MaxRois, ps32Proposals, u32NumAfterFilter);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "getRPNresult error, error code is %d", s32Ret);
    return HI_SUCCESS;
    /******************** end of FasterRCNN RPN **********************/
}

HI_S32 HI_NodePlugin_Compute(const HI_NodePlugin_Operand_S *pstInputs, HI_U32 u32InputNum,
    HI_NodePlugin_Operand_S *pstOutputs, HI_U32 u32Outputs, HI_NodePlugin_NodeParam_S* pstHyperParam, HI_NodePlugin_NodeParam_S* pstTrainingParam)
{
    HI_U32 assist_mem_size = 0;
    PROPOSAL_PARA_S para;
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32RoisNum = 0;
    HI_S32 *pInputArray[2];
    HI_NodePlugin_Operand_S pstTmpInputs[2];
    HI_NodePlugin_Operand_S stTmp;
    HI_RUNTIME_MEM_S stMem;
    memset(&para, 0x0, sizeof(PROPOSAL_PARA_S));
    SAMPLE_CHK_RETURN(u32InputNum != 2, HI_FAILURE, "proposal inputs number error,the corrent number is %u\n", u32InputNum);
    memcpy(pstTmpInputs, pstInputs, sizeof(HI_NodePlugin_Operand_S) * u32InputNum);
    if (pstTmpInputs[0].stShape.s32C > pstTmpInputs[1].stShape.s32C)
    {
        memcpy(&stTmp, &pstTmpInputs[0], sizeof(HI_NodePlugin_Operand_S));
        memcpy(&pstTmpInputs[0], &pstTmpInputs[1], sizeof(HI_NodePlugin_Operand_S));
        memcpy(&pstTmpInputs[1], &stTmp, sizeof(HI_NodePlugin_Operand_S));
    }

    for (HI_U32 i = 0; i < u32InputNum; i++)
    {
        para.model_info.astReportNodeInfo[i].u32ConvHeight = pstTmpInputs[i].stShape.s32H;
        para.model_info.astReportNodeInfo[i].u32ConvMapNum = pstTmpInputs[i].stShape.s32C;
        para.model_info.astReportNodeInfo[i].u32ConvStride = pstTmpInputs[i].u32Stride;
        para.model_info.astReportNodeInfo[i].u32ConvWidth = pstTmpInputs[i].stShape.s32W;
    }

    getParam(pstHyperParam->pParam, &para);
    assist_mem_size = GetRFCNAssistMemSize(&para);
    stMem.u32Size = assist_mem_size;
    s32Ret = SAMPLE_AllocMem(&stMem, HI_TRUE);
    SAMPLE_CHK_RETURN(s32Ret != HI_SUCCESS, HI_FAILURE, "SAMPLE_AllocMem error, return value :%d\n", s32Ret);

    s32Ret = SAMPLE_FlushCache(&stMem);
    SAMPLE_CHK_GOTO(s32Ret != HI_SUCCESS, COMPUTE_FLUSHCACHE_ERROR, "SAMPLE_FlushMem error, return value :%d\n", s32Ret);
    pInputArray[0] = (HI_S32*)((HI_UL)pstTmpInputs[0].u64Offset);
    pInputArray[1] = (HI_S32*)((HI_UL)pstTmpInputs[1].u64Offset);

    s32Ret = HI_MPI_SVP_NNIE_WK_CNN_FASTER_RPN_Ref(pInputArray,
        2,
        &para,
        (HI_U32*)((HI_UL)stMem.u64VirAddr),
        (HI_S32*)((HI_UL)(pstOutputs[0].u64Offset)),
        &u32RoisNum);

    SAMPLE_FreeMem(&stMem);
    pstOutputs[0].stShape.s32H = u32RoisNum;
    return s32Ret;
COMPUTE_FLUSHCACHE_ERROR:
    SAMPLE_FreeMem(&stMem);
    return HI_FAILURE;
}

HI_S32 HI_NodePlugin_getNodeType(HI_CHAR pszNodeType[], const HI_U32 u32Length)
{
    HI_U32 u32ProposalLength = strlen("Proposal");
    SAMPLE_CHK_RETURN((u32Length < sizeof("Proposal")), HI_FAILURE, "param u32Length is too small");
    strncpy(pszNodeType, "Proposal", u32ProposalLength);
    pszNodeType[u32ProposalLength] = '\0';
    return HI_SUCCESS;
}
