#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sample_mutex_ops.h>
#include "hi_runtime_api.h"
#include "string.h"
#include "sample_log.h"
#include "sample_memory_ops.h"
#ifdef ON_BOARD
#include "mpi_sys.h"
#include "mpi_vb.h"
#else
#include "back/hi_comm_svp.h"
#include "back/hi_nnie.h"
#include "back/mpi_nnie.h"
#endif
#include "math.h"
#include "sample_save_blob.h"
#include "sample_resize_roi.h"
#include "sample_data_utils.h"
#include "sample_runtime_define.h"
#include "sample_cop_param.h"
#include "sample_data_utils.h"
#include "sample_ive_image.h"
#include "sample_runtime_group_tracker.h"

#define RFCN_DST_BLOB_NUM (3)

#define MAX_TARGETS_NUM (2)
#define TRACKER_MEM_NUM (3)
#define IMAGE_WIDTH (352)
#define IMAGE_HIGHT (288)

#ifndef TRACKER_DEBUG
//#define TRACKER_DEBUG
#endif

#ifndef CLASSIFY
//#define CLASSIFY
#endif

#define MODEL_RFCN_DIR RESOURCE_DIR
#define MODEL_ALEXNET_DIR RESOURCE_DIR
#define MODEL_GOTURN_DIR RESOURCE_DIR
#define IMAGE_DIR RESOURCE_DIR
#define CONFIG_DIR "./"


typedef struct tagTRACKER_BLOBS
{
    HI_RUNTIME_GROUP_SRC_BLOB_ARRAY_S stGroupSrc;
    HI_RUNTIME_GROUP_DST_BLOB_ARRAY_S stGroupDst;
    // just include group's input
    HI_RUNTIME_GROUP_BLOB_S astInputs[1];
    HI_RUNTIME_BLOB_S stInputBlobImage;
    // include all model/connector 's output(top)
    HI_RUNTIME_GROUP_BLOB_S astOutputs[13];
    HI_RUNTIME_BLOB_S astOutputBlobRfcnAfter[1];
    HI_RUNTIME_BLOB_S astOutputBlobFeatureExtra1[1];
    HI_RUNTIME_BLOB_S astOutputBlobGoTurnPre[1];
    HI_RUNTIME_BLOB_S astOutputBlobFeatureExtra2[1];
    HI_RUNTIME_BLOB_S astOutputBlobCompare[2];
#ifdef CLASSIFY
    HI_RUNTIME_BLOB_S astOutputBlobResize1[1];
    HI_RUNTIME_BLOB_S astOutputBlobResize2[1];
    HI_RUNTIME_BLOB_S astOutputBlobResize3[1];
    HI_RUNTIME_BLOB_S astOutputBlobAlexNet1[1];
    HI_RUNTIME_BLOB_S astOutputBlobAlexNet2[1];
    HI_RUNTIME_BLOB_S astOutputBlobAlexNet3[1];
#endif
}TRACKER_BLOBS_S;


static TRACKER_BLOBS_S *s_gstTrackerBlobs;
static HI_RUNTIME_GROUP_INFO_S s_stGroupInfo;
static HI_BOOL s_bFinish = HI_FALSE;
static HI_U64 g_u64CurFrame;
static SAMPLE_COND s_finishCond;
static SAMPLE_MUTEX s_finishMutex;
static HI_BOOL s_bIsFirstFrame = HI_TRUE;

const HI_DOUBLE kScaleFactor = 10;
static HI_VOID unscaleBbox(HI_S32 width, HI_S32 height, BondingBox_s *pstBbox)
{
    pstBbox->x1 /= kScaleFactor;
    pstBbox->x2 /= kScaleFactor;
    pstBbox->y1 /= kScaleFactor;
    pstBbox->y2 /= kScaleFactor;

    pstBbox->x1 *= width;
    pstBbox->x2 *= width;
    pstBbox->y1 *= height;
    pstBbox->y2 *= height;
}

static HI_VOID uncenterBbox(HI_S32 width, HI_S32 height, BondingBox_s* pstLocation,
    HI_DOUBLE edgeX, HI_DOUBLE edgeY,
    BondingBox_s* pstEstimateBbox, BondingBox_s* pstDstBbox)
{
    pstDstBbox->x1 = max(0.0, pstEstimateBbox->x1 + pstLocation->x1 - edgeX);
    pstDstBbox->y1 = max(0.0, pstEstimateBbox->y1 + pstLocation->y1 - edgeY);
    pstDstBbox->x2 = min(width, pstEstimateBbox->x2 + pstLocation->x1 - edgeX);
    pstDstBbox->y2 = min(height, pstEstimateBbox->y2 + pstLocation->y1 - edgeY);
}

static HI_S32 Connector_RfcnPre(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_RfcnPre \n");
#endif
    resizeBlob(&pstConnectorSrc->pstBlobs[0], &pstConnectorDst->pstBlobs[0]);
    return HI_SUCCESS;
}

static HI_S32 Connector_RfcnAfter(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_RfcnAfter \n");
#endif
    HI_S32 as32ResultROI[300 * SVP_WK_PROPOSAL_WIDTH] = { 0 };
    HI_U32 u32ResultROICnt = 0;
    HI_U32 *pu32DstAddr = HI_NULL;
    HI_CHAR aszResizeFileName[16];

    memset(aszResizeFileName, 0x0, sizeof(aszResizeFileName));
    HI_S32 s32Ret = SAMPLE_DATA_GetRoiResultFromOriginSize(SAMPLE_RUNTIME_MODEL_TYPE_RFCN,
                           &pstConnectorSrc->pstBlobs[1], // bbox
                           &pstConnectorSrc->pstBlobs[2], // score
                           &pstConnectorSrc->pstBlobs[0], // proposal
                           800,
                           600,
                           as32ResultROI,
                           &u32ResultROICnt);
    SAMPLE_CHK_RETURN((s32Ret != HI_SUCCESS), HI_FAILURE, "SAMPLE_DATA_GetRoiResult failed\n");
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("roi cnt: %u\n", u32ResultROICnt);
#endif
    pu32DstAddr = (HI_U32*)((HI_UL)(pstConnectorDst->pstBlobs[0]).u64VirAddr);
    // only support MAX_TARGETS_NUM targets
    pstConnectorDst->pstBlobs[0].u32Num = MAX_TARGETS_NUM;

    for (HI_U32 i = 0; i < pstConnectorDst->pstBlobs[0].u32Num; i++)
    {
       pu32DstAddr[0] = as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH];
       pu32DstAddr[1] = as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH + 1];
       pu32DstAddr[2] = as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH + 2];
       pu32DstAddr[3] = as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH + 3];
       pu32DstAddr += (pstConnectorDst->pstBlobs[0]).u32Stride / sizeof(HI_U32);
    }
    return HI_SUCCESS;
}

static HI_S32 Connector_FeatureExtra1(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_FeatureExtra1 \n");
#endif
    HI_U8 *pu8SrcAddr = (HI_U8*)((HI_UL)pstConnectorSrc->pstBlobs[0].u64VirAddr);
    HI_U8 *pu8DstAddr = (HI_U8*)((HI_UL)pstConnectorDst->pstBlobs[0].u64VirAddr);
    HI_U32 u32CopySize = pstConnectorSrc->pstBlobs[0].u32Num * pstConnectorSrc->pstBlobs[0].unShape.stWhc.u32Chn *
                         pstConnectorSrc->pstBlobs[0].unShape.stWhc.u32Height * pstConnectorSrc->pstBlobs[0].u32Stride;
    memcpy(pu8DstAddr, pu8SrcAddr, u32CopySize);
    return HI_SUCCESS;
}

static HI_S32 Connector_GoTurnPre(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_GoTurnPre srdID[%lld]\n", srcId);
#endif
    if ((3 != pstConnectorSrc->u32BlobNum) || (3 != pstConnectorDst->u32BlobNum))
    {
        SAMPLE_LOG_PRINT("Invalid input number [%u] or output number [%u]\n", pstConnectorSrc->u32BlobNum, pstConnectorDst->u32BlobNum);
        return HI_FAILURE;
    }

    if (HI_TRUE == s_bIsFirstFrame)
    {
        pstConnectorDst->pstBlobs[0].u32Num = 0;
        pstConnectorDst->pstBlobs[1].u32Num = 0;
        pstConnectorDst->pstBlobs[2].u32Num = 0;
        s_bIsFirstFrame = HI_FALSE;
        return HI_SUCCESS;
    }

    HI_RUNTIME_BLOB_S *pstImageBlob = &pstConnectorSrc->pstBlobs[0];
    HI_RUNTIME_BLOB_S *pstTargetBlob = &pstConnectorSrc->pstBlobs[1];
    HI_RUNTIME_BLOB_S *pstTargetsBboxBlob = &pstConnectorSrc->pstBlobs[2];
    HI_RUNTIME_BLOB_S *pstImageCropBlob = &pstConnectorDst->pstBlobs[0];
    HI_RUNTIME_BLOB_S *pstTargetCropBlob = &pstConnectorDst->pstBlobs[1];
    HI_RUNTIME_BLOB_S *pstLocationEdge = &pstConnectorDst->pstBlobs[2];
    TRACKER_LOCATION_S astLocation[MAX_TARGETS_NUM];
    HI_S32 as32EdgeX[MAX_TARGETS_NUM];
    HI_S32 as32EdgeY[MAX_TARGETS_NUM];
    HI_S32 as32RegionX[MAX_TARGETS_NUM];
    HI_S32 as32RegionY[MAX_TARGETS_NUM];

    memset(astLocation, 0x0, sizeof(astLocation));
    memset(as32EdgeX, 0x0, sizeof(as32EdgeX));
    memset(as32EdgeY, 0x0, sizeof(as32EdgeY));
    memset(as32RegionX, 0x0, sizeof(as32RegionX));
    memset(as32RegionY, 0x0, sizeof(as32RegionY));

#ifdef TRACKER_DEBUG
    /* 111 98  137 200 */
    HI_S32* ps32Tmp = (HI_S32*)((HI_UL)(pstTargetsBboxBlob->u64VirAddr));
    SAMPLE_LOG_PRINT("target 1 (x1,y1,x2,y2) = (%d, %d, %d, %d)\n", *(ps32Tmp), *(ps32Tmp + 1), *(ps32Tmp + 2), *(ps32Tmp + 3));
    ps32Tmp = (HI_S32*)((HI_UL)(pstTargetsBboxBlob->u64VirAddr)
        + pstTargetsBboxBlob->u32Stride * pstTargetsBboxBlob->unShape.stWhc.u32Chn * pstTargetsBboxBlob->unShape.stWhc.u32Height);
    SAMPLE_LOG_PRINT("target 2 (x1,y1,x2,y2) = (%d, %d, %d, %d)\n", *(ps32Tmp), *(ps32Tmp + 1), *(ps32Tmp + 2), *(ps32Tmp + 3));

#endif

    cropPadBlob(pstImageBlob,
        pstTargetsBboxBlob,
        pstImageCropBlob,
        &astLocation[0],
        &as32EdgeX[0],
        &as32EdgeY[0],
        &as32RegionX[0],
        &as32RegionY[0]);

    HI_S32 *ps32Dst = HI_NULL;
    HI_U32 u32OffSet = pstLocationEdge->unShape.stWhc.u32Chn * pstLocationEdge->unShape.stWhc.u32Height * pstLocationEdge->u32Stride;
    for (HI_U32 i = 0; i < pstLocationEdge->u32Num; i++)
    {
        ps32Dst = (HI_S32*)((HI_UL)(pstLocationEdge->u64VirAddr) + i * u32OffSet);
        *ps32Dst = astLocation[i].x1;
        *(ps32Dst + 1) = astLocation[i].y1;
        *(ps32Dst + 2) = astLocation[i].x2;
        *(ps32Dst + 3) = astLocation[i].y2;
        *(ps32Dst + 4) = as32EdgeX[i];
        *(ps32Dst + 5) = as32EdgeY[i];
        *(ps32Dst + 6) = as32RegionX[i];
        *(ps32Dst + 7) = as32RegionY[i];
    }

    cropPadBlob(pstTargetBlob,
        pstTargetsBboxBlob,
        pstTargetCropBlob,
        &astLocation[0],
        &as32EdgeX[0],
        &as32EdgeY[0],
        &as32RegionX[0],
        &as32RegionY[0]);

    return HI_SUCCESS;
}

static HI_S32 Connector_GoTurnAfter(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_GoTurnAfter \n");
#endif
    if ((2 != pstConnectorSrc->u32BlobNum) || (1 != pstConnectorDst->u32BlobNum))
    {
        SAMPLE_LOG_PRINT("Invalid input number [%u] or output number [%u]\n", pstConnectorSrc->u32BlobNum, pstConnectorDst->u32BlobNum);
        return HI_FAILURE;
    }
    HI_S32 *ps32SrcEstimate = NULL;
    HI_S32 *ps32SrcLocationEdge = NULL;
    HI_S32 *ps32Dst = NULL;
    HI_RUNTIME_BLOB_S *pstSrcEstimateBlob = &pstConnectorSrc->pstBlobs[0];
    HI_RUNTIME_BLOB_S *pstSrcLocationEdgeBlob = &pstConnectorSrc->pstBlobs[1];
    HI_RUNTIME_BLOB_S *pstDstBlob = &pstConnectorDst->pstBlobs[0];
    HI_U32 u32SrcEstimateOffSet = pstSrcEstimateBlob->u32Stride * pstSrcEstimateBlob->unShape.stWhc.u32Height * pstSrcEstimateBlob->unShape.stWhc.u32Chn;
    HI_U32 u32SrcLocationEdgeOffSet = pstSrcLocationEdgeBlob->u32Stride * pstSrcLocationEdgeBlob->unShape.stWhc.u32Height * pstSrcLocationEdgeBlob->unShape.stWhc.u32Chn;
    HI_U32 u32DstBboxOffSet = pstDstBlob->u32Stride * pstDstBlob->unShape.stWhc.u32Height * pstDstBlob->unShape.stWhc.u32Chn;
    BondingBox_s stEstimate;
    BondingBox_s stLocation;
    BondingBox_s stDstBbox;
    HI_DOUBLE dEdgeX;
    HI_DOUBLE dEdgeY;
    HI_S32 s32RegionW;
    HI_S32 s32RegionH;

    for (HI_S32 i = 0; i < pstSrcEstimateBlob->u32Num; i++)
    {
        ps32SrcEstimate = (HI_S32 *)((HI_UL)(pstSrcEstimateBlob->u64VirAddr) + i * u32SrcEstimateOffSet);
        stEstimate.x1 = (HI_DOUBLE)(*ps32SrcEstimate) / 4096;
        stEstimate.y1 = (HI_DOUBLE)(*(ps32SrcEstimate + 1)) / 4096;
        stEstimate.x2 = (HI_DOUBLE)(*(ps32SrcEstimate + 2)) / 4096;
        stEstimate.y2 = (HI_DOUBLE)(*(ps32SrcEstimate + 3)) / 4096;

        ps32SrcLocationEdge = (HI_S32 *)((HI_UL)(pstSrcLocationEdgeBlob->u64VirAddr) + i * u32SrcLocationEdgeOffSet);

        stLocation.x1 = (HI_DOUBLE)(*ps32SrcLocationEdge) / 4096;
        stLocation.y1 = (HI_DOUBLE)(*(ps32SrcLocationEdge + 1)) / 4096;
        stLocation.x2 = (HI_DOUBLE)(*(ps32SrcLocationEdge + 2)) / 4096;
        stLocation.y2 = (HI_DOUBLE)(*(ps32SrcLocationEdge + 3)) / 4096;
        dEdgeX = (HI_DOUBLE)(*(ps32SrcLocationEdge + 4)) / 4096;
        dEdgeY = (HI_DOUBLE)(*(ps32SrcLocationEdge + 5)) / 4096;
        s32RegionW = *(ps32SrcLocationEdge + 6);
        s32RegionH = *(ps32SrcLocationEdge + 7);

        SAMPLE_LOG_PRINT("estimate %d : (%f, %f)(%f, %f)\n", i, stEstimate.x1, stEstimate.y1, stEstimate.x2, stEstimate.y2);
        SAMPLE_LOG_PRINT("--------- %d, %d, %d, %d\n", s32RegionW, s32RegionH, IMAGE_WIDTH, IMAGE_HIGHT);
        unscaleBbox(s32RegionW, s32RegionH, &stEstimate);
        uncenterBbox(IMAGE_WIDTH, IMAGE_HIGHT, &stLocation, dEdgeX, dEdgeY, &stEstimate, &stDstBbox);

        ps32Dst = (HI_S32 *)((HI_UL)(pstDstBlob->u64VirAddr) + i * u32DstBboxOffSet);
        *(ps32Dst) = stDstBbox.x1;
        *(ps32Dst + 1) = stDstBbox.y1;
        *(ps32Dst + 2) = stDstBbox.x2;
        *(ps32Dst + 3) = stDstBbox.y2;
    }
    return HI_SUCCESS;
}

static HI_S32 Connector_FeatureExtra2(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_FeatureExtra2 \n");
#endif
    HI_U8 *pu8SrcAddr = (HI_U8*)((HI_UL)(pstConnectorSrc->pstBlobs[0].u64VirAddr));
    HI_U8 *pu8DstAddr = (HI_U8*)((HI_UL)(pstConnectorDst->pstBlobs[0].u64VirAddr));
    HI_U32 u32CopySize = pstConnectorSrc->pstBlobs[0].u32Num * pstConnectorSrc->pstBlobs[0].unShape.stWhc.u32Chn
                         * pstConnectorSrc->pstBlobs[0].unShape.stWhc.u32Height * pstConnectorSrc->pstBlobs[0].u32Stride;
    memcpy(pu8DstAddr, pu8SrcAddr, u32CopySize);

    return HI_SUCCESS;
}

static HI_S32 copyToTarget(HI_RUNTIME_BLOB_S* pDstTarget,
    HI_RUNTIME_BLOB_S* pOrigImageBlob, HI_RUNTIME_BLOB_S* pSrcBlob, HI_BOOL bIsScale, HI_U64 srcId)
{
    HI_U32 u32ResultCnt = 0;
    HI_U8* pu8DstAddr = HI_NULL;
    HI_U32* pu32TmpDstAddr = HI_NULL;
    HI_U32* pu32TmpSrcAddr = HI_NULL;
    HI_S32 as32ResultROI[300 * 4] = { 0 };
    HI_CHAR aszResizeFileName[16];
    memset(aszResizeFileName, 0x0, sizeof(aszResizeFileName));
    pu8DstAddr = (HI_U8*)((HI_UL)(pDstTarget->u64VirAddr));
    u32ResultCnt = pSrcBlob->u32Num;

    pu32TmpDstAddr = (HI_U32*)pu8DstAddr;
    pu32TmpSrcAddr = (HI_U32*)((HI_UL)(pSrcBlob->u64VirAddr));
    for(HI_U32 i = 0; i < u32ResultCnt; i++)
    {
        if (HI_TRUE == bIsScale)
        {
            as32ResultROI[i * 4] =(HI_U32)(pu32TmpSrcAddr[i * 4] * ((HI_FLOAT)pOrigImageBlob->unShape.stWhc.u32Width / 800));
            as32ResultROI[i * 4 + 1] =(HI_U32)(pu32TmpSrcAddr[i * 4 + 1] * ((HI_FLOAT)pOrigImageBlob->unShape.stWhc.u32Height / 600));
            as32ResultROI[i * 4 + 2] =(HI_U32)(pu32TmpSrcAddr[i * 4 + 2] * ((HI_FLOAT)pOrigImageBlob->unShape.stWhc.u32Width / 800));
            as32ResultROI[i * 4 + 3] =(HI_U32)(pu32TmpSrcAddr[i * 4 + 3] * ((HI_FLOAT)pOrigImageBlob->unShape.stWhc.u32Height / 600));
        }
        else
        {
            as32ResultROI[i * 4] =(HI_U32)(pu32TmpSrcAddr[i * 4]);
            as32ResultROI[i * 4 + 1] =(HI_U32)(pu32TmpSrcAddr[i * 4 + 1]);
            as32ResultROI[i * 4 + 2] =(HI_U32)(pu32TmpSrcAddr[i * 4 + 2]);
            as32ResultROI[i * 4 + 3] =(HI_U32)(pu32TmpSrcAddr[i * 4 + 3]);

        }
        pu32TmpDstAddr[i * 4] = as32ResultROI[i * 4];
        pu32TmpDstAddr[i * 4 + 1] = as32ResultROI[i * 4 + 1];
        pu32TmpDstAddr[i * 4 + 2] = as32ResultROI[i * 4 + 2];
        pu32TmpDstAddr[i * 4 + 3] = as32ResultROI[i * 4 + 3];
#ifdef TRACKER_DEBUG
        SAMPLE_LOG_PRINT("==========cord %d,%d,%d,%d========\n", as32ResultROI[i * 4], as32ResultROI[i * 4 + 1], as32ResultROI[i * 4 + 2], as32ResultROI[i * 4 + 3]);
#endif
    }

    snprintf(aszResizeFileName, sizeof(aszResizeFileName), "%lld_out", srcId);
    drawImageRect(aszResizeFileName, pOrigImageBlob, as32ResultROI, u32ResultCnt, 4);

    return HI_SUCCESS;
}
static HI_S32 Connector_Compare(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
#ifdef TRACKER_DEBUG
    SAMPLE_LOG_PRINT("\n Connector_Compare \n");
#endif
    HI_S32 s32Ret = HI_FAILURE;
    if (0 == pstConnectorSrc->pstBlobs[2].u32Num)
    {
        s32Ret = copyToTarget(&(pstConnectorDst->pstBlobs[1]),
                              &(pstConnectorSrc->pstBlobs[0]), &(pstConnectorSrc->pstBlobs[1]),
                              HI_TRUE, srcId);
    }
    else
    {
        s32Ret = copyToTarget(&(pstConnectorDst->pstBlobs[1]),
                              &(pstConnectorSrc->pstBlobs[0]), &(pstConnectorSrc->pstBlobs[2]),
                              HI_FALSE, srcId);
    }

    HI_U8 *pu8SrcAddr = (HI_U8*)((HI_UL)(pstConnectorSrc->pstBlobs[0].u64VirAddr));
    HI_U8 *pu8DstAddr = (HI_U8*)((HI_UL)(pstConnectorDst->pstBlobs[0].u64VirAddr));
    HI_U32 u32CopySize = pstConnectorSrc->pstBlobs[0].u32Num * pstConnectorSrc->pstBlobs[0].unShape.stWhc.u32Chn
                         * pstConnectorSrc->pstBlobs[0].unShape.stWhc.u32Height * pstConnectorSrc->pstBlobs[0].u32Stride;
    memcpy(pu8DstAddr, pu8SrcAddr, u32CopySize);

    return s32Ret;
}
#ifdef CLASSIFY
static HI_S32 Connector_Resize1(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
    return HI_SUCCESS;
}

static HI_S32 Connector_Resize2(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
    return HI_SUCCESS;
}

static HI_S32 Connector_Resize3(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 srcId, HI_VOID* pParam)
{
    return HI_SUCCESS;
}
#endif
static HI_S32 SAMPLE_RUNTIME_LoadModelGroup_RFCNGoTrunAlex(
        const HI_CHAR* pcModelFileRFCN,
        const HI_CHAR* pcModelFileGoturn,
        const HI_CHAR* pcModelFileAlex,
        HI_RUNTIME_WK_INFO_S* pstWkInfo,
        HI_PROPOSAL_Param_S *pProposalParam,
        HI_RUNTIME_GROUP_HANDLE* phGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;

    HI_RUNTIME_COP_ATTR_S stProposalAttr = {0};
    HI_RUNTIME_CONNECTOR_ATTR_S astConnectorAttr[10] = {0};
    HI_CHAR acConfig[4096] = {0};

    memset(&stProposalAttr, 0, sizeof(HI_RUNTIME_COP_ATTR_S));
    memset(&astConnectorAttr[0], 0, sizeof(astConnectorAttr));

    // wk mem
    strncpy(pstWkInfo[0].acModelName, "rfcn", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileRFCN, &pstWkInfo[0].stWKMemory);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileRFCN);

    strncpy(pstWkInfo[1].acModelName, "go_turn", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileGoturn, &pstWkInfo[1].stWKMemory);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileGoturn);
#ifdef CLASSIFY
    strncpy(pstWkInfo[2].acModelName, "alexnet_1", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileAlex, &pstWkInfo[2].stWKMemory);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileAlex);

    strncpy(pstWkInfo[3].acModelName, "alexnet_2", MAX_NAME_LEN);
    memcpy(&pstWkInfo[3].stWKMemory, &pstWkInfo[2].stWKMemory, sizeof(HI_RUNTIME_MEM_S));

    strncpy(pstWkInfo[4].acModelName, "alexnet_3", MAX_NAME_LEN);
    memcpy(&pstWkInfo[4].stWKMemory, &pstWkInfo[2].stWKMemory, sizeof(HI_RUNTIME_MEM_S));
#endif
    // cop param
    strncpy(stProposalAttr.acModelName, "rfcn", MAX_NAME_LEN);
    strncpy(stProposalAttr.acCopName, "proposal", MAX_NAME_LEN);
    stProposalAttr.u32ConstParamSize = sizeof(HI_PROPOSAL_Param_S);
    s32Ret = createRFCNCopParam(1, &stProposalAttr, pProposalParam);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createRFCNCopParam failed!\n");

    // conector
    strncpy(astConnectorAttr[0].acName, "rfcn_pre", MAX_NAME_LEN);
    astConnectorAttr[0].pParam = NULL;
    astConnectorAttr[0].pConnectorFun = Connector_RfcnPre;

    strncpy(astConnectorAttr[1].acName, "rfcn_after", MAX_NAME_LEN);
    astConnectorAttr[1].pParam = NULL;
    astConnectorAttr[1].pConnectorFun = Connector_RfcnAfter;

    strncpy(astConnectorAttr[2].acName, "feature_extra_1", MAX_NAME_LEN);
    astConnectorAttr[2].pParam = NULL;
    astConnectorAttr[2].pConnectorFun = Connector_FeatureExtra1;

    strncpy(astConnectorAttr[3].acName, "go_turn_pre", MAX_NAME_LEN);
    astConnectorAttr[3].pParam = NULL;
    astConnectorAttr[3].pConnectorFun = Connector_GoTurnPre;

    strncpy(astConnectorAttr[4].acName, "go_turn_after", MAX_NAME_LEN);
    astConnectorAttr[4].pParam = NULL;
    astConnectorAttr[4].pConnectorFun = Connector_GoTurnAfter;

    strncpy(astConnectorAttr[5].acName, "feature_extra_2", MAX_NAME_LEN);
    astConnectorAttr[5].pParam = NULL;
    astConnectorAttr[5].pConnectorFun = Connector_FeatureExtra2;

    strncpy(astConnectorAttr[6].acName, "compare", MAX_NAME_LEN);
    astConnectorAttr[6].pParam = NULL;
    astConnectorAttr[6].pConnectorFun = Connector_Compare;
#ifdef CLASSIFY
    strncpy(astConnectorAttr[7].acName, "resize_1", MAX_NAME_LEN);
    astConnectorAttr[7].pParam = NULL;
    astConnectorAttr[7].pConnectorFun = Connector_Resize1;

    strncpy(astConnectorAttr[8].acName, "resize_2", MAX_NAME_LEN);
    astConnectorAttr[8].pParam = NULL;
    astConnectorAttr[8].pConnectorFun = Connector_Resize2;

    strncpy(astConnectorAttr[9].acName, "resize_3", MAX_NAME_LEN);
    astConnectorAttr[9].pParam = NULL;
    astConnectorAttr[9].pConnectorFun = Connector_Resize3;
#endif
    SAMPLE_RUNTIME_ReadConfig(CONFIG_DIR"rfcn_goturn_alexnet_tracker.modelgroup", acConfig, 4096);
    acConfig[sizeof(acConfig) - 1] = '\0';

    s_stGroupInfo.stWKsInfo.u32WKNum = 2;
#ifdef CLASSIFY
    s_stGroupInfo.stWKsInfo.u32WKNum = 5;
#endif
    s_stGroupInfo.stWKsInfo.pstAttrs = &pstWkInfo[0];

    s_stGroupInfo.stCopsAttr.u32CopNum = 1;
    s_stGroupInfo.stCopsAttr.pstAttrs = &stProposalAttr;
    s_stGroupInfo.stConnectorsAttr.u32ConnectorNum = 7;
#ifdef CLASSIFY
    s_stGroupInfo.stConnectorsAttr.u32ConnectorNum = 10;
#endif
    s_stGroupInfo.stConnectorsAttr.pstAttrs = &astConnectorAttr[0];

    s32Ret = HI_SVPRT_RUNTIME_LoadModelGroup(acConfig, &s_stGroupInfo, phGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_SVPRT_RUNTIME_LoadModelGroupSync failed!\n");

    SAMPLE_LOG_PRINT("LoadGroup succ\n");

FAIL_0:
    return s32Ret;
}

static HI_S32 forward_finish(
        HI_RUNTIME_FORWARD_STATUS_CALLBACK_E enEvent,
        HI_RUNTIME_GROUP_HANDLE hGroupHandle,
        HI_U64 u64FrameId,
        HI_RUNTIME_GROUP_DST_BLOB_ARRAY_PTR pstDstBlobs)
{
    SAMPLE_LOG_PRINT("forward finish: %lld\n", u64FrameId);
    SAMPLE_CHK_GOTO(HI_RUNTIME_FORWARD_STATUS_SUCC != enEvent, FAIL_0, "Forward finish failed[%u]!\n", enEvent);
    SAMPLE_CHK_GOTO(u64FrameId != g_u64CurFrame + 1, FAIL_0, "Forward finish with error frameId[%lld], current frameId[%lld]\n", u64FrameId, g_u64CurFrame);

    SAMPLE_MUTEX_Lock(&s_finishMutex);
    g_u64CurFrame = u64FrameId;
    SAMPLE_COND_Broadcast(&s_finishCond);
    SAMPLE_MUTEX_Unlock(&s_finishMutex);
    return HI_SUCCESS;

FAIL_0:
    SAMPLE_MUTEX_Lock(&s_finishMutex);
    s_bFinish = HI_TRUE;
    g_u64CurFrame = u64FrameId;
    SAMPLE_COND_Broadcast(&s_finishCond);
    SAMPLE_MUTEX_Unlock(&s_finishMutex);
    return HI_SUCCESS;
}

HI_VOID blobFree(TRACKER_BLOBS_S *pstBlobs)
{
    HI_RUNTIME_MEM_S stMem = { 0 };

    for (HI_U32 i = 0; i < TRACKER_MEM_NUM; i++)
    {
        stMem.u64PhyAddr = pstBlobs[i].stInputBlobImage.u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].stInputBlobImage.u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobRfcnAfter[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobRfcnAfter[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobFeatureExtra1[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobFeatureExtra1[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobGoTurnPre[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobGoTurnPre[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobFeatureExtra2[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobFeatureExtra2[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobCompare[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobCompare[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobCompare[1].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobCompare[1].u64VirAddr;
        SAMPLE_FreeMem(&stMem);
#ifdef CLASSIFY
        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobResize1[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobResize1[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobResize2[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobResize2[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobResize3[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobResize3[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobAlexNet1[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobAlexNet1[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobAlexNet2[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobAlexNet2[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);

        stMem.u64PhyAddr = pstBlobs[i].astOutputBlobAlexNet3[0].u64PhyAddr;
        stMem.u64VirAddr = pstBlobs[i].astOutputBlobAlexNet3[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);
#endif
    }
    SAMPLE_FREE(pstBlobs);
}

static HI_S32 blobsInit(TRACKER_BLOBS_S **ppstBlobs)
{
    HI_S32 s32Ret = HI_SUCCESS;
    *ppstBlobs = (TRACKER_BLOBS_S *)malloc(sizeof(TRACKER_BLOBS_S) * TRACKER_MEM_NUM);
    if (HI_NULL == *ppstBlobs)
    {
        return HI_FAILURE;
    }
    memset((HI_CHAR *)*ppstBlobs, 0, sizeof(TRACKER_BLOBS_S));
    TRACKER_BLOBS_S *pstBlobs = *ppstBlobs;
    for (HI_U32 i = 0; i < TRACKER_MEM_NUM; i++)
    {
        pstBlobs[i].stGroupSrc.u32BlobNum = 1;
        pstBlobs[i].stGroupSrc.pstBlobs = &pstBlobs[i].astInputs[0];

        pstBlobs[i].stGroupDst.u32BlobNum = 6;
#ifdef CLASSIFY
        pstBlobs[i].stGroupDst.u32BlobNum = 12;
#endif
        pstBlobs[i].stGroupDst.pstBlobs = &pstBlobs[i].astOutputs[0];

        pstBlobs[i].astInputs[0].pstBlob = &pstBlobs[i].stInputBlobImage;
        strncpy(pstBlobs[i].astInputs[0].acOwnerName, "", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astInputs[0].acBlobName, "image", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[0].pstBlob = &pstBlobs[i].astOutputBlobRfcnAfter[0];
        strncpy(pstBlobs[i].astOutputs[0].acOwnerName, "rfcn_after", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[0].acBlobName, "detection_targets", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[1].pstBlob = &pstBlobs[i].astOutputBlobFeatureExtra1[0];
        strncpy(pstBlobs[i].astOutputs[1].acOwnerName, "feature_extra_1", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[1].acBlobName, "targets_feature_d", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[2].pstBlob = &pstBlobs[i].astOutputBlobGoTurnPre[0];
        strncpy(pstBlobs[i].astOutputs[2].acOwnerName, "go_turn_pre", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[2].acBlobName, "location_edge", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[3].pstBlob = &pstBlobs[i].astOutputBlobFeatureExtra2[0];
        strncpy(pstBlobs[i].astOutputs[3].acOwnerName, "feature_extra_2", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[3].acBlobName, "targets_feature_t", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[4].pstBlob = &pstBlobs[i].astOutputBlobCompare[0];
        strncpy(pstBlobs[i].astOutputs[4].acOwnerName, "compare", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[4].acBlobName, "targets", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[5].pstBlob = &pstBlobs[i].astOutputBlobCompare[1];
        strncpy(pstBlobs[i].astOutputs[5].acOwnerName, "compare", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[5].acBlobName, "targets_bbox", MAX_NAME_LEN);
#ifdef CLASSIFY
        pstBlobs[i].astOutputs[6].pstBlob = &pstBlobs[i].astOutputBlobResize1[0];
        strncpy(pstBlobs[i].astOutputs[6].acOwnerName, "res0ize_1", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[6].acBlobName, "targets_resize", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[7].pstBlob = &pstBlobs[i].astOutputBlobResize2[0];
        strncpy(pstBlobs[i].astOutputs[7].acOwnerName, "resize_2", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[7].acBlobName, "targets_resize", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[8].pstBlob = &pstBlobs[i].astOutputBlobResize3[0];
        strncpy(pstBlobs[i].astOutputs[8].acOwnerName, "resize_3", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[8].acBlobName, "targets_resize", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[9].pstBlob = &pstBlobs[i].astOutputBlobAlexNet1[0];
        strncpy(pstBlobs[i].astOutputs[9].acOwnerName, "alexnet_1", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[9].acBlobName, "prob", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[10].pstBlob = &pstBlobs[i].astOutputBlobAlexNet2[0];
        strncpy(pstBlobs[i].astOutputs[10].acOwnerName, "alexnet_2", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[10].acBlobName, "prob", MAX_NAME_LEN);

        pstBlobs[i].astOutputs[11].pstBlob = &pstBlobs[i].astOutputBlobAlexNet3[0];
        strncpy(pstBlobs[i].astOutputs[11].acOwnerName, "alexnet_3", MAX_NAME_LEN);
        strncpy(pstBlobs[i].astOutputs[11].acBlobName, "prob", MAX_NAME_LEN);
#endif

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].stInputBlobImage, HI_RUNTIME_BLOB_TYPE_U8, 1, IMAGE_WIDTH, IMAGE_HIGHT, 3, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobRfcnAfter[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 4, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobFeatureExtra1[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 4, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobGoTurnPre[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 8, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobFeatureExtra2[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 4, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobCompare[0], HI_RUNTIME_BLOB_TYPE_U8, 1, IMAGE_WIDTH, IMAGE_HIGHT, 3, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobCompare[1], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 4, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");
#ifdef CLASSIFY
        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobResize1[0], HI_RUNTIME_BLOB_TYPE_U8, MAX_TARGETS_NUM, 227, 227, 3, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobResize2[0], HI_RUNTIME_BLOB_TYPE_U8, MAX_TARGETS_NUM, 227, 227, 3, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobResize3[0], HI_RUNTIME_BLOB_TYPE_U8, MAX_TARGETS_NUM, 227, 227, 3, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobAlexNet1[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 1000, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob alex net dst failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobAlexNet2[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 1000, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob alex net dst failed!\n");

        s32Ret = SAMPLE_RUNTIME_SetBlob(&pstBlobs[i].astOutputBlobAlexNet3[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_TARGETS_NUM, 1, 1, 1000, ALIGN_16);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob alex net dst failed!\n");
#endif
    }

    return s32Ret;
FAIL_0:
    blobFree(pstBlobs);
    return s32Ret;
}

static HI_VOID blobsUpdate(TRACKER_BLOBS_S *pstBlobs)
{
    pstBlobs->stInputBlobImage.u32Num = 1;

    pstBlobs->astOutputBlobRfcnAfter[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobFeatureExtra1[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobGoTurnPre[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobFeatureExtra2[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobCompare[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobCompare[1].u32Num = MAX_TARGETS_NUM;
#ifdef CLASSIFY
    pstBlobs->astOutputBlobResize1[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobResize2[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobResize3[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobAlexNet1[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobAlexNet2[0].u32Num = MAX_TARGETS_NUM;

    pstBlobs->astOutputBlobAlexNet3[0].u32Num = MAX_TARGETS_NUM;
#endif

}

static HI_S32 SAMPLE_RUNTIME_ForwardGroup_RFCNGoTurnAlex(const HI_CHAR* pcSrcFile, HI_RUNTIME_GROUP_HANDLE hGroupHandle, HI_U64 u64StartFrame, HI_U64 u64EndFrame)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32MemIndex = 0;
    HI_U32 i = 0;
    HI_CHAR acFileName[256] = { 0 };

    if (HI_SUCCESS != blobsInit(&s_gstTrackerBlobs))
    {
        SAMPLE_LOG_PRINT("Malloc fail!\n");
        return HI_FAILURE;
    }
    g_u64CurFrame = u64StartFrame - 1;

    for (i = 0; i < TRACKER_MEM_NUM;)
    {
        snprintf(acFileName, sizeof(acFileName), pcSrcFile, i + u64StartFrame);
        s32Ret = SAMPLE_RUNTIME_ReadSrcFile(acFileName, &(s_gstTrackerBlobs[i].stInputBlobImage));
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, END_0, "Read image fail, frameNum = %lld\n", i + u64StartFrame);

        s32Ret = HI_SVPRT_RUNTIME_ForwardGroupASync(hGroupHandle, &(s_gstTrackerBlobs[i].stGroupSrc), &(s_gstTrackerBlobs[i].stGroupDst), i + u64StartFrame, forward_finish);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, END_0, "Forward fail, srcId = %lld\n", i + u64StartFrame - 1);
        i++;
        SAMPLE_CHK_GOTO(i + u64StartFrame - 1 >= u64EndFrame, END_0, "Reach end frame, srcId = %lld\n", i + u64StartFrame - 1);
    }
    {

        SAMPLE_MUTEX_Lock(&s_finishMutex);

        while (!s_bFinish)
        {
            SAMPLE_COND_Wait(&s_finishCond, &s_finishMutex);
            if ((g_u64CurFrame == u64EndFrame) || (i + u64StartFrame > u64EndFrame))
            {
                SAMPLE_LOG_PRINT("Finish forward from %lld to %lld\n", u64StartFrame, u64EndFrame);
                break;
            }
            else
            {
                u32MemIndex = (g_u64CurFrame - u64StartFrame) % TRACKER_MEM_NUM;
                blobsUpdate(&s_gstTrackerBlobs[u32MemIndex]);
                snprintf(acFileName, sizeof(acFileName), pcSrcFile, i + u64StartFrame);
                s32Ret = SAMPLE_RUNTIME_ReadSrcFile(acFileName, &(s_gstTrackerBlobs[u32MemIndex].stInputBlobImage));
                if(HI_SUCCESS != s32Ret)
                {
                    SAMPLE_LOG_PRINT("Read image fail, frameNum = %lld\n", i + u64StartFrame);
                    SAMPLE_MUTEX_Unlock(&s_finishMutex);
                    goto END_0;
                }

                s32Ret = HI_SVPRT_RUNTIME_ForwardGroupASync(hGroupHandle, &(s_gstTrackerBlobs[u32MemIndex].stGroupSrc), &(s_gstTrackerBlobs[u32MemIndex].stGroupDst), i + u64StartFrame, forward_finish);
                if(HI_SUCCESS != s32Ret)
                {
                    SAMPLE_LOG_PRINT("Forward fail, srcId = %lld\n", g_u64CurFrame + TRACKER_MEM_NUM);
                    SAMPLE_MUTEX_Unlock(&s_finishMutex);
                    goto END_0;
                }

                i++;
            }
        }

        SAMPLE_MUTEX_Unlock(&s_finishMutex);
    }

END_0:
    {
        printf("wait for finish 1, current frame[%lld], end frame[%lld]\n", g_u64CurFrame, i + u64StartFrame - 1);

        SAMPLE_MUTEX_Lock(&s_finishMutex);

        while (g_u64CurFrame < i + u64StartFrame - 1)
        {
            printf("wait for finish, current frame[%lld], end frame[%lld]\n", g_u64CurFrame, i + u64StartFrame - 1);
            SAMPLE_COND_Wait(&s_finishCond, &s_finishMutex);
        }
        SAMPLE_MUTEX_Unlock(&s_finishMutex);
    }

    blobFree(s_gstTrackerBlobs);
    return s32Ret;
}

HI_S32 SAMPLE_Model_Group_RFCN_GOTURN_ALEXNET(HI_U64 u64StartFrame, HI_U64 u64EndFrame)
{
    HI_S32 s32Ret = HI_SUCCESS;

#ifdef ON_BOARD
    const HI_CHAR* pcRuntimeModelRFCN = MODEL_RFCN_DIR"/runtime_rfcn_resnet50_inst.wk";
    const HI_CHAR* pcRuntimeModelAlexNet = MODEL_ALEXNET_DIR"/runtime_alexnet_no_group_inst.wk";
    const HI_CHAR* pcRuntimeModelGoTurn = MODEL_GOTURN_DIR"/runtime_tracker_goturn_inst.wk";
#else
    const HI_CHAR* pcRuntimeModelRFCN = MODEL_RFCN_DIR"/runtime_rfcn_resnet50_func.wk";
    const HI_CHAR* pcRuntimeModelAlexNet = MODEL_ALEXNET_DIR"/runtime_alexnet_no_group_func.wk";
    const HI_CHAR* pcRuntimeModelGoTurn = MODEL_GOTURN_DIR"/runtime_tracker_goturn_func.wk";
#endif
    const HI_CHAR* pcSrcFile = IMAGE_DIR"/%08lld.bgr";

    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);

    HI_RUNTIME_GROUP_HANDLE hGroupHandle;
    HI_PROPOSAL_Param_S copParam[1] = {0};
    HI_RUNTIME_WK_INFO_S astWkInfo[5] = {0};

    memset(&copParam[0], 0, sizeof(copParam));
    memset(&astWkInfo[0], 0, sizeof(astWkInfo));

    SAMPLE_MUTEX_Init(&s_finishMutex);
    SAMPLE_COND_Init(&s_finishCond);
    SAMPLE_LOG_PRINT("\n============================= rfcn&goturn&alexnet group begin ================================\n");
    s32Ret = HI_SVPRT_RUNTIME_Init("cpu_task_affinity:2 cpu_task_affinity:3", NULL);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_1, "HI_SVPRT_RUNTIME_Init failed!\n");

    s32Ret = SAMPLE_RUNTIME_LoadModelGroup_RFCNGoTrunAlex(pcRuntimeModelRFCN,
            pcRuntimeModelGoTurn,
            pcRuntimeModelAlexNet,
            astWkInfo,
            copParam,
            &hGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelGroup_RFCNGoTrunAlex failed!\n");

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Load]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = SAMPLE_RUNTIME_ForwardGroup_RFCNGoTurnAlex(pcSrcFile, hGroupHandle, u64StartFrame, u64EndFrame);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "SAMPLE_RUNTIME_ForwardGroup_Alexnet error\n");
    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Forward total]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = HI_SVPRT_RUNTIME_UnloadModelGroup(hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "HI_SVPRT_RUNTIME_UnloadModelGroup error\n");


FAIL_0:
    HI_SVPRT_RUNTIME_DeInit();

    releaseRfcnAndFrcnnCopParam(1, copParam);

    SAMPLE_FreeMem(&(astWkInfo[0].stWKMemory));
    SAMPLE_FreeMem(&(astWkInfo[1].stWKMemory));

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Total]===== TIME SPEND: %ld ms =====\n", spend);
FAIL_1:
    SAMPLE_MUTEX_Deinit(&s_finishMutex);
    SAMPLE_COND_Deinit(&s_finishCond);
    SAMPLE_LOG_PRINT("SAMPLE_Model_Group_RFCN_GOTURN_ALEXNET result %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", s32Ret);
    return s32Ret;
}
