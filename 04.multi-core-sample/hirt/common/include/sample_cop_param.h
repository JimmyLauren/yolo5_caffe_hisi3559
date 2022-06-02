#ifndef __SAMPLE_COP_PARAM_H
#define __SAMPLE_COP_PARAM_H
#include "back/hi_nnie.h"
#include "hi_runtime_comm.h"

#ifdef __cplusplus
extern "C"
{
#endif

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

HI_S32 createFasterrcnnCopParam(HI_U32 u32ParamNum, HI_RUNTIME_COP_ATTR_S* pCopParam, HI_PROPOSAL_Param_S* pPluginParam);
HI_S32 createRFCNCopParam(HI_U32 u32ParamNum, HI_RUNTIME_COP_ATTR_S* pCopParam, HI_PROPOSAL_Param_S* pPluginParam);
HI_VOID releaseRfcnAndFrcnnCopParam(HI_U32 u32ParamNum, HI_PROPOSAL_Param_S* pCopParam);

#ifdef __cplusplus
}
#endif

#endif
