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
#include "sample_runtime_group_rfcnalex.h"
#include "sample_cop_param.h"
#include "sample_data_utils.h"

#define FRCNN_DST_BLOB_NUM 3

#define MODEL_RFCN_DIR RESOURCE_DIR
#define MODEL_ALEXNET_DIR RESOURCE_DIR
#define IMAGE_DIR RESOURCE_DIR
#define CONFIG_DIR "./"

static HI_U32 s_u32RoiNum = 0;
static HI_S32 Connector_RFCNToAlexNet(HI_RUNTIME_SRC_BLOB_ARRAY_PTR pstConnectorSrc, HI_RUNTIME_DST_BLOB_ARRAY_PTR pstConnectorDst, HI_U64 u64SrcId, HI_VOID* pParam)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_S32 as32ResultROI[300 * SVP_WK_PROPOSAL_WIDTH] = { 0 };
    HI_U32 u32ResultROICnt = 0;
    HI_U32 u32TempCnt = 0;
    HI_U32 *pu32RoiNum = (HI_U32*)pParam;
#if DEBUG
    HI_CHAR aszResizeFileName[16];
#endif

    long spend;
    struct timespec start, end;
    clock_gettime(0, &start);
#if DEBUG
    HI_CHAR* pcOutName = HI_NULL;
    HI_U32 i = 0;

    for (i = 0; i < 4; i++)
    {
        if (0 == i) { pcOutName = (HI_CHAR*)"data"; }

        if (1 == i) { pcOutName = (HI_CHAR*)"proposal"; }

        if (2 == i) { pcOutName = (HI_CHAR*)"cls_prob_reshape"; }

        if (3 == i) { pcOutName = (HI_CHAR*)"bbox_pred_reshape"; }

        printDebugData(pcOutName, pstConnectorSrc->pstBlobs[i].u64VirAddr, 10);
    }

#endif
    s32Ret = SAMPLE_DATA_GetRoiResult(SAMPLE_RUNTIME_MODEL_TYPE_RFCN,
                           &pstConnectorSrc->pstBlobs[2], // bbox
                           &pstConnectorSrc->pstBlobs[3], // score
                           &pstConnectorSrc->pstBlobs[1], // proposal
                           &pstConnectorSrc->pstBlobs[0], //orig image
                           as32ResultROI,
                           &u32ResultROICnt);
    SAMPLE_CHK_RETURN((s32Ret != HI_SUCCESS), HI_FAILURE, "SAMPLE_DATA_GetRoiResult failed\n");
    printf("roi cnt: %u\n", u32ResultROICnt);
    u32TempCnt = u32ResultROICnt;
    *pu32RoiNum = u32ResultROICnt;
#if DEBUG

    for (i = 0; i < u32ResultROICnt; i++)
    {
        printf("ROI[%d]: x1=%d y1=%d x2=%d y2=%d\n", i,
               as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH],
               as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH + 1],
               as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH + 2],
               as32ResultROI[i * SVP_WK_PROPOSAL_WIDTH + 3]
              );
    }

#endif

    while (u32TempCnt > SAMPLE_IVE_RESIZE_BATCH_MAX)
    {
        resizeROI(&pstConnectorSrc->pstBlobs[0],
                  &as32ResultROI[(u32ResultROICnt - u32TempCnt) * SVP_WK_PROPOSAL_WIDTH],
                  SVP_WK_PROPOSAL_WIDTH,
                  SAMPLE_IVE_RESIZE_BATCH_MAX,
                  &pstConnectorDst->pstBlobs[0],
                  u32ResultROICnt - u32TempCnt);
        u32TempCnt -= SAMPLE_IVE_RESIZE_BATCH_MAX;
    }

    resizeROI(&pstConnectorSrc->pstBlobs[0],
              &as32ResultROI[(u32ResultROICnt - u32TempCnt) * SVP_WK_PROPOSAL_WIDTH],
              SVP_WK_PROPOSAL_WIDTH,
              (HI_U16)u32TempCnt,
              &pstConnectorDst->pstBlobs[0],
              u32ResultROICnt - u32TempCnt);

    pstConnectorDst->pstBlobs[0].u32Num = u32ResultROICnt;

#if DEBUG

    for (i = 0; i < u32ResultROICnt; i++)
    {
        snprintf(aszResizeFileName, sizeof(aszResizeFileName), "%d_ra", i);
        saveBlob(aszResizeFileName, &pstConnectorDst->pstBlobs[0], i);
    }

    FILE* fp = NULL;
    HI_RUNTIME_BLOB_S* pstBlob = &pstConnectorDst->pstBlobs[0];
    fp = fopen("rfcn_out.bgr", "w");
    SAMPLE_CHK_RETURN((HI_NULL == fp), HI_FAILURE, "open rfcn_out.bgr failed\n");
    HI_U32 c, h;
    HI_U8* pu8vir = NULL;
    pu8vir = (HI_U8*)((HI_UL)(pstConnectorDst->pstBlobs[0].u64VirAddr));

    for (c = 0; c < pstBlob->unShape.stWhc.u32Chn; c++)
        for (h = 0; h < pstBlob->unShape.stWhc.u32Height; h++)
        {
            fwrite(pu8vir, 1, pstBlob->unShape.stWhc.u32Width * sizeof(HI_U8), fp);
            pu8vir += pstBlob->u32Stride;
        }

    fclose(fp);
#endif
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("\n[Connector]===== TIME SPEND: %ld ms =====", spend);

    return HI_SUCCESS;
}

static HI_RUNTIME_GROUP_INFO_S s_stGroupInfo;

static HI_S32 SAMPLE_RUNTIME_LoadModelGroup_RFCNAlex(const HI_CHAR* pcModelFileRFCN,
        HI_RUNTIME_WK_INFO_S* pstWkInfo,
        const HI_CHAR* pcModelFileAlex,
        HI_PROPOSAL_Param_S *pProposalParam,
        HI_RUNTIME_GROUP_HANDLE* phGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;

    HI_RUNTIME_COP_ATTR_S stProposalAttr = {0};
    HI_RUNTIME_CONNECTOR_ATTR_S stConnectorAttr = {0};
    HI_CHAR acConfig[1024] = {0};

    memset(&stProposalAttr, 0, sizeof(HI_RUNTIME_COP_ATTR_S));
    memset(&stConnectorAttr, 0, sizeof(HI_RUNTIME_CONNECTOR_ATTR_S));
    memset(acConfig, 0, sizeof(acConfig));
    // wk mem
    strncpy(pstWkInfo[0].acModelName, "rfcn", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileRFCN, &pstWkInfo[0].stWKMemory);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileRFCN);
    strncpy(pstWkInfo[1].acModelName, "alexnet", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileAlex, &pstWkInfo[1].stWKMemory);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileAlex);

    // cop param
    strncpy(stProposalAttr.acModelName, "rfcn", MAX_NAME_LEN);
    strncpy(stProposalAttr.acCopName, "proposal", MAX_NAME_LEN);
    stProposalAttr.u32ConstParamSize = sizeof(HI_PROPOSAL_Param_S);
    s32Ret = createRFCNCopParam(1, &stProposalAttr, pProposalParam);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createRFCNCopParam failed!\n");

    // conector
    strncpy(stConnectorAttr.acName, "rfcn_conn_alexnet", MAX_NAME_LEN);
    stConnectorAttr.pParam = &s_u32RoiNum;
    stConnectorAttr.pConnectorFun = Connector_RFCNToAlexNet;

    SAMPLE_RUNTIME_ReadConfig(CONFIG_DIR"rfcn_alexnet.modelgroup", acConfig, 1024);
    acConfig[1023] = '\0';

    s_stGroupInfo.stWKsInfo.u32WKNum = 2;
    s_stGroupInfo.stWKsInfo.pstAttrs = &pstWkInfo[0];

    s_stGroupInfo.stCopsAttr.u32CopNum = 1;
    s_stGroupInfo.stCopsAttr.pstAttrs = &stProposalAttr;

    s_stGroupInfo.stConnectorsAttr.u32ConnectorNum = 1;
    s_stGroupInfo.stConnectorsAttr.pstAttrs = &stConnectorAttr;

    s32Ret = HI_SVPRT_RUNTIME_LoadModelGroup(acConfig, &s_stGroupInfo, phGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_SVPRT_RUNTIME_LoadModelGroupSync failed!\n");

    SAMPLE_LOG_PRINT("LoadGroup succ\n");

FAIL_0:
    return s32Ret;
}

static HI_S32 SAMPLE_RUNTIME_SetGroupSrc_RFCNAlex(
        const HI_CHAR* pcSrcFile,
        HI_RUNTIME_BLOB_S* pstInputBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;

    //Rfcn src
    s32Ret = SAMPLE_RUNTIME_SetBlob(&pstInputBlobs[0], HI_RUNTIME_BLOB_TYPE_U8, 1, 800, 600, 3, ALIGN_16);
    SAMPLE_CHK_RETURN(HI_SUCCESS != s32Ret, s32Ret, "SAMPLE_RUNTIME_SetBlob faster rcnn failed!\n");

    s32Ret = SAMPLE_RUNTIME_ReadSrcFile(pcSrcFile, &(pstInputBlobs[0]));
    SAMPLE_CHK_RETURN(HI_SUCCESS != s32Ret, s32Ret, "ReadSrcFile failed!\n");
    return s32Ret;
}

static HI_S32 SAMPLE_RUNTIME_SetGroupDst_RFCNAlex(HI_RUNTIME_BLOB_S* pstDstAlexBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;

    //Alex net dst
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstAlexBlobs[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, MAX_ROI_NUM, 1, 1, 1000, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob alex net dst failed!\n");

FAIL_0:
    // MMZ free outside this function
    return s32Ret;
}

static HI_BOOL s_bFinish = HI_FALSE;
static SAMPLE_COND s_finishCond;
static SAMPLE_MUTEX s_finishMutex;

static HI_S32 rfcn_alexnet_forward_finish(
        HI_RUNTIME_FORWARD_STATUS_CALLBACK_E enEvent,
        HI_RUNTIME_GROUP_HANDLE hGroupHandle,
        HI_U64 u64FrameId,
        HI_RUNTIME_GROUP_DST_BLOB_ARRAY_PTR pstDstBlobs)
{
    printf("forward finish: %lld\n", u64FrameId);
    SAMPLE_MUTEX_Lock(&s_finishMutex);
    s_bFinish = HI_TRUE;
    SAMPLE_COND_Broadcast(&s_finishCond);
    SAMPLE_MUTEX_Unlock(&s_finishMutex);
    return HI_SUCCESS;
}

static HI_S32 SAMPLE_RUNTIME_ForwardGroup_RFCNAlex(const HI_CHAR* pcSrcFile, HI_RUNTIME_GROUP_HANDLE hGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_GROUP_SRC_BLOB_ARRAY_S stGroupSrc;
    HI_RUNTIME_GROUP_DST_BLOB_ARRAY_S stGroupDst;
    // just include group's input
    HI_RUNTIME_GROUP_BLOB_S astInputBlob[1];
    // include all model/connector 's output(top)
    HI_RUNTIME_GROUP_BLOB_S astOutputBlob[1];
    HI_RUNTIME_MEM_S stMem;

    HI_RUNTIME_BLOB_S astInputBlobs[1];
    HI_RUNTIME_BLOB_S astDstAlexBlobs[1];

    memset(&astInputBlob[0], 0, sizeof(astInputBlob));
    memset(&astOutputBlob[0], 0, sizeof(astOutputBlob));

    stGroupSrc.u32BlobNum = 1;
    stGroupSrc.pstBlobs = &astInputBlob[0];

    stGroupDst.u32BlobNum = 1;
    stGroupDst.pstBlobs = &astOutputBlob[0];

    strncpy(stGroupSrc.pstBlobs[0].acOwnerName, "", MAX_NAME_LEN);
    strncpy(stGroupSrc.pstBlobs[0].acBlobName, "data", MAX_NAME_LEN);
    stGroupSrc.pstBlobs[0].pstBlob = &astInputBlobs[0];

    strncpy(stGroupDst.pstBlobs[0].acOwnerName, "alexnet", MAX_NAME_LEN);
    strncpy(stGroupDst.pstBlobs[0].acBlobName, "prob", MAX_NAME_LEN);
    stGroupDst.pstBlobs[0].pstBlob = &astDstAlexBlobs[0];

    memset(astInputBlobs, 0x0, sizeof(astInputBlobs));
    memset(astDstAlexBlobs, 0x0, sizeof(astDstAlexBlobs));

#if PERFORMANCE_TEST
    long spend;
    struct timespec start, end;
#endif

    s32Ret = SAMPLE_RUNTIME_SetGroupSrc_RFCNAlex(pcSrcFile, astInputBlobs);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetGroupSrc_FrcnnAlex failed!\n");

    s32Ret = SAMPLE_RUNTIME_SetGroupDst_RFCNAlex(astDstAlexBlobs);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetGroupDst_FrcnnAlex failed!\n");
#if PERFORMANCE_TEST
    clock_gettime(0, &start);
#endif

    s32Ret = HI_SVPRT_RUNTIME_ForwardGroupASync(hGroupHandle, &stGroupSrc, &stGroupDst, 0, rfcn_alexnet_forward_finish);

    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "Forward fail");

    {
        SAMPLE_MUTEX_Lock(&s_finishMutex);

        while (!s_bFinish)
        {
            SAMPLE_COND_Wait(&s_finishCond, &s_finishMutex);
        }
        SAMPLE_MUTEX_Unlock(&s_finishMutex);
    }

    {
        SAMPLE_MUTEX_Lock(&s_finishMutex);
        s_bFinish = HI_FALSE;
        SAMPLE_MUTEX_Unlock(&s_finishMutex);
    }

#if PERFORMANCE_TEST
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 * 1000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("\n[Forward]===== TIME SPEND: %ldms, %ldus =====\n\n", spend/1000, spend);
#endif

    stGroupDst.pstBlobs[0].pstBlob->u32Num = s_u32RoiNum;
    s32Ret = SAMPLE_RUNTIME_Cnn_TopN_Output(stGroupDst.pstBlobs[0].pstBlob, 1);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_Cnn_TopN_Output failed!\n");
FAIL_0:
    if (0 != astInputBlobs[0].u64PhyAddr)
    {
        stMem.u64PhyAddr = astInputBlobs[0].u64PhyAddr;
        stMem.u64VirAddr = astInputBlobs[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);
    }
    if (0 != astDstAlexBlobs[0].u64PhyAddr)
    {
        stMem.u64PhyAddr = astDstAlexBlobs[0].u64PhyAddr;
        stMem.u64VirAddr = astDstAlexBlobs[0].u64VirAddr;
        SAMPLE_FreeMem(&stMem);
    }
    return s32Ret;
}

// fasterRcnn->connector->alexNet
HI_S32 SAMPLE_Model_Group_RFCNAlexNet()
{
    HI_S32 s32Ret = HI_SUCCESS;
#ifdef ON_BOARD
    const HI_CHAR* pcRuntimeModelRFCN = MODEL_RFCN_DIR"/runtime_rfcn_resnet50_inst.wk";
    const HI_CHAR* pcRuntimeModelAlexNet = MODEL_ALEXNET_DIR"/runtime_alexnet_no_group_inst.wk";
#else
    const HI_CHAR* pcRuntimeModelRFCN = MODEL_RFCN_DIR"/runtime_rfcn_resnet50_func.wk";
    const HI_CHAR* pcRuntimeModelAlexNet = MODEL_ALEXNET_DIR"/runtime_alexnet_no_group_func.wk";
#endif
    const HI_CHAR* pcSrcFile = IMAGE_DIR"/horse_dog_car_person_600x800.bgr";

    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);

    HI_RUNTIME_GROUP_HANDLE hGroupHandle;
    HI_PROPOSAL_Param_S copParam[1];
    HI_RUNTIME_WK_INFO_S astWkInfo[2];

    memset(&copParam[0], 0, sizeof(copParam));
    memset(&astWkInfo[0], 0, sizeof(astWkInfo));

    SAMPLE_MUTEX_Init(&s_finishMutex);
    SAMPLE_COND_Init(&s_finishCond);
    printf("\n============================= rfcn & alex net group begin ================================\n");
    s32Ret = HI_SVPRT_RUNTIME_Init(CPU_TASK_AFFINITY, NULL);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_1, "HI_SVPRT_RUNTIME_Init failed!\n");

    s32Ret = SAMPLE_RUNTIME_LoadModelGroup_RFCNAlex(pcRuntimeModelRFCN,
            astWkInfo,
            pcRuntimeModelAlexNet,
            copParam,
            &hGroupHandle);
    printf("============loadmodel hGroupHandle[%p]\n", hGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelGroup failed!\n");

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("\n[Load]===== TIME SPEND: %ld ms =====\n", spend);

    printf("============forward hGroupHandle[%p]\n", hGroupHandle);
    s32Ret = SAMPLE_RUNTIME_ForwardGroup_RFCNAlex(pcSrcFile, hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "SAMPLE_RUNTIME_ForwardGroup_Alexnet error\n");

    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    printf("\n[Forward total]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = HI_SVPRT_RUNTIME_UnloadModelGroup(hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "HI_SVPRT_RUNTIME_UnloadModelGroup error\n");


FAIL_0:
    HI_SVPRT_RUNTIME_DeInit();

    releaseRfcnAndFrcnnCopParam(1, copParam);

    SAMPLE_FreeMem(&(astWkInfo[0].stWKMemory));
    SAMPLE_FreeMem(&(astWkInfo[1].stWKMemory));

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    printf("\n[Total]===== TIME SPEND: %ld ms =====\n", spend);
FAIL_1:
    SAMPLE_MUTEX_Deinit(&s_finishMutex);
    SAMPLE_COND_Deinit(&s_finishCond);
    SAMPLE_LOG_PRINT("SAMPLE_Model_Group_RFCNAlexNet result %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", s32Ret);
    return s32Ret;
}
