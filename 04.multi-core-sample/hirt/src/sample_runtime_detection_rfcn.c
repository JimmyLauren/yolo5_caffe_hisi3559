#include <stdio.h>
#include <stdlib.h>
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
#include "sample_runtime_detection_rfcn.h"
#include "sample_cop_param.h"

#define MODEL_DIR RESOURCE_DIR
#define IMAGE_DIR RESOURCE_DIR
#define CONFIG_DIR "./"

HI_S32 SAMPLE_RUNTIME_LoadModelGroup_RFCN(const HI_CHAR* pcModelFileRFCN,
        HI_RUNTIME_WK_INFO_S* pstWkInfo,
        HI_PROPOSAL_Param_S* copParam,
        HI_RUNTIME_GROUP_HANDLE* phGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_GROUP_INFO_S stGroupInfo = {0};
    HI_CHAR acConfig[1024] = {0};

    memset(acConfig, 0, sizeof(acConfig));
    memset(&stGroupInfo, 0, sizeof(HI_RUNTIME_GROUP_INFO_S));

    strncpy(pstWkInfo[0].acModelName, "rfcn", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileRFCN, &(pstWkInfo[0].stWKMemory));
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0,
                                   "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileRFCN);
    stGroupInfo.stWKsInfo.u32WKNum = 1;
    stGroupInfo.stWKsInfo.pstAttrs = &(pstWkInfo[0]);

    HI_RUNTIME_COP_ATTR_S stCopAttr;
    memset(&stCopAttr, 0, sizeof(HI_RUNTIME_COP_ATTR_S));
    strncpy(stCopAttr.acModelName, "rfcn", MAX_NAME_LEN);
    strncpy(stCopAttr.acCopName, "proposal", MAX_NAME_LEN);
    stCopAttr.u32ConstParamSize = sizeof(HI_PROPOSAL_Param_S);
    s32Ret = createRFCNCopParam(1, &stCopAttr, copParam);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "createRFCNCopParam failed!\n");
    stGroupInfo.stCopsAttr.u32CopNum = 1;
    stGroupInfo.stCopsAttr.pstAttrs = &stCopAttr;
    SAMPLE_RUNTIME_ReadConfig(CONFIG_DIR"rfcn.modelgroup", acConfig, 1024);
    acConfig[1023] = '\0';

    s32Ret = HI_SVPRT_RUNTIME_LoadModelGroup(acConfig, &stGroupInfo, phGroupHandle);
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0,
                                   "HI_SVPRT_RUNTIME_LoadModelGroup error\n");

    sample_debug("LoadGroup succ, group handle[%p]\n", pstGroup->pHiRTInternal);

    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetGroupSrc_RFCN(const HI_CHAR* pcSrcFile,
        HI_RUNTIME_BLOB_S* pstSrcRFCNBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;

    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstSrcRFCNBlobs[0], HI_RUNTIME_BLOB_TYPE_U8, 1, 800, 600, 3, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob alexnet failed!\n");

    s32Ret = SAMPLE_RUNTIME_ReadSrcFile(pcSrcFile, &(pstSrcRFCNBlobs[0]));
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_ReadSrcFile failed!\n");

    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetGroupDst_RFCN(HI_RUNTIME_BLOB_S* pstDstRFCNBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;

    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstRFCNBlobs[0],
            HI_RUNTIME_BLOB_TYPE_S32, 1, 4, 300, 1, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob[0] rfcn failed!\n");

    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstRFCNBlobs[1],
            HI_RUNTIME_BLOB_TYPE_S32, 300, 21, 1, 1, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob[1] rfcn failed!\n");

    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstRFCNBlobs[2],
            HI_RUNTIME_BLOB_TYPE_S32, 300, 8, 1, 1, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob[2] rfcn failed!\n");

    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_ForwardGroup_RFCN(const HI_CHAR* pcSrcFile, HI_RUNTIME_GROUP_HANDLE hGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
#if DEBUG
    HI_CHAR* pcOutName = HI_NULL;
#endif
    HI_S32 as32ResultROI[300 * SVP_WK_PROPOSAL_WIDTH] = { 0 };
    HI_U32 u32ResultROICnt = 0;
    HI_RUNTIME_GROUP_SRC_BLOB_ARRAY_S stGroupSrcBlob;
    HI_RUNTIME_GROUP_DST_BLOB_ARRAY_S stGroupDstBlob;
    HI_RUNTIME_GROUP_BLOB_S astInputBlob[1];
    HI_RUNTIME_GROUP_BLOB_S astOutputBlob[3];
    HI_RUNTIME_MEM_S stMem = {0};
    memset(&astInputBlob[0], 0, sizeof(astInputBlob));
    memset(&astOutputBlob[0], 0, sizeof(astOutputBlob));

    HI_RUNTIME_BLOB_S stSrc[1];
    HI_RUNTIME_BLOB_S stDst[3];
    memset(&stSrc[0], 0, sizeof(HI_RUNTIME_BLOB_S));
    memset(&stDst[0], 0, sizeof(HI_RUNTIME_BLOB_S));
    memset(&stDst[1], 0, sizeof(HI_RUNTIME_BLOB_S));
    memset(&stDst[2], 0, sizeof(HI_RUNTIME_BLOB_S));

    // src
    stGroupSrcBlob.u32BlobNum = 1;
    stGroupSrcBlob.pstBlobs = &(astInputBlob[0]);

    strncpy(astInputBlob[0].acOwnerName, "", MAX_NAME_LEN);
    astInputBlob[0].acOwnerName[MAX_NAME_LEN] = 0;
    strncpy(astInputBlob[0].acBlobName, "data", MAX_NAME_LEN);
    astInputBlob[0].acBlobName[MAX_NAME_LEN] = 0;
    astInputBlob[0].pstBlob = &stSrc[0];

    // dst
    stGroupDstBlob.u32BlobNum = 3;
    stGroupDstBlob.pstBlobs = &(astOutputBlob[0]);

    strncpy(astOutputBlob[0].acOwnerName, "rfcn", MAX_NAME_LEN);
    astOutputBlob[0].acOwnerName[MAX_NAME_LEN] = 0;
    strncpy(astOutputBlob[0].acBlobName, "proposal", MAX_NAME_LEN);
    astOutputBlob[0].acBlobName[MAX_NAME_LEN] = 0;
    astOutputBlob[0].pstBlob = &stDst[0];

    strncpy(astOutputBlob[1].acOwnerName, "rfcn", MAX_NAME_LEN);
    astOutputBlob[1].acOwnerName[MAX_NAME_LEN] = 0;
    strncpy(astOutputBlob[1].acBlobName, "cls_prob_reshape", MAX_NAME_LEN);
    astOutputBlob[1].acBlobName[MAX_NAME_LEN] = 0;
    astOutputBlob[1].pstBlob = &stDst[1];

    strncpy(astOutputBlob[2].acOwnerName, "rfcn", MAX_NAME_LEN);
    astOutputBlob[2].acOwnerName[MAX_NAME_LEN] = 0;
    strncpy(astOutputBlob[2].acBlobName, "bbox_pred_reshape", MAX_NAME_LEN);
    astOutputBlob[2].acBlobName[MAX_NAME_LEN] = 0;
    astOutputBlob[2].pstBlob = &stDst[2];


    s32Ret = SAMPLE_RUNTIME_SetGroupSrc_RFCN(pcSrcFile, stSrc);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetGroupSrc_FrcnnAlex failed!\n");

    s32Ret = SAMPLE_RUNTIME_SetGroupDst_RFCN(stDst);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetGroupDst_FrcnnAlex failed!\n");
#if PERFORMANCE_TEST
    long spend;
    struct timespec start, end;
    clock_gettime(0, &start);
#endif

    s32Ret = HI_SVPRT_RUNTIME_ForwardGroupSync(hGroupHandle, &stGroupSrcBlob, &stGroupDstBlob, 0);
#if PERFORMANCE_TEST
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 * 1000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("\n\n[Forward]===== TIME SPEND: %ldms, %ldus =====\n\n", spend/1000, spend);
#endif
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "HI_SVPRT_RUNTIME_ForwardGroupSync failed!\n");

    s32Ret = SAMPLE_DATA_GetRoiResult(SAMPLE_RUNTIME_MODEL_TYPE_RFCN, &stDst[1], &stDst[2], &stDst[0],
            &stSrc[0], as32ResultROI, &u32ResultROICnt);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_DATA_GetRoiResult failed!\n");
    SAMPLE_LOG_PRINT("====Pic: %s====\nroi cnt: %d\n", pcSrcFile, u32ResultROICnt);
    for (HI_S32 i = 0; i < u32ResultROICnt; i++)
    {
        if (as32ResultROI[i*6+4]>0.8*4096)
        {
            SAMPLE_LOG_PRINT("%d %d %d %d %lf(%d)\n",
                    as32ResultROI[i*6], as32ResultROI[i*6+1],
                    as32ResultROI[i*6+2], as32ResultROI[i*6+3],
                    (HI_DOUBLE)(as32ResultROI[i*6+4]*((HI_DOUBLE)1.0)/4096), as32ResultROI[i*6+4]);
        }
    }
    SAMPLE_LOG_PRINT("====detection info end====\n");

    drawImageRect("rfcn", &stSrc[0], as32ResultROI, u32ResultROICnt, 6);

#if DEBUG
#ifdef ON_BOARD
    drawImageRect("rfcn", &stSrc[0], as32ResultROI, u32ResultROICnt, 6);
#endif
    for (int i = 0; i < 3; i++)
    {
        if (0 == i) { pcOutName = (HI_CHAR*)"proposal"; }

        if (1 == i) { pcOutName = (HI_CHAR*)"cls_prob_reshape"; }

        if (2 == i) { pcOutName = (HI_CHAR*)"bbox_pred_reshape"; }

        printDebugData(pcOutName, stDst[i].u64VirAddr, 10);
    }

    printf("proposal bbox num is: %d\n", stDst[0].unShape.stWhc.u32Height);
#endif

FAIL_0:

    stMem.u64PhyAddr = stSrc[0].u64PhyAddr;
    stMem.u64VirAddr = stSrc[0].u64VirAddr;
    SAMPLE_FreeMem(&stMem);

    stMem.u64PhyAddr = stDst[0].u64PhyAddr;
    stMem.u64VirAddr = stDst[0].u64VirAddr;
    SAMPLE_FreeMem(&stMem);

    stMem.u64PhyAddr = stDst[1].u64PhyAddr;
    stMem.u64VirAddr = stDst[1].u64VirAddr;
    SAMPLE_FreeMem(&stMem);

    stMem.u64PhyAddr = stDst[2].u64PhyAddr;
    stMem.u64VirAddr = stDst[2].u64VirAddr;
    SAMPLE_FreeMem(&stMem);
    return s32Ret;
}

HI_S32 SAMPLE_RFCN()
{
#ifdef ON_BOARD
    const HI_CHAR* pcRuntimeModelName = MODEL_DIR"/runtime_rfcn_resnet50_inst.wk";
#else
    const HI_CHAR* pcRuntimeModelName = MODEL_DIR"/runtime_rfcn_resnet50_func.wk";
#endif
    const HI_CHAR* pcSrcFile = IMAGE_DIR"/horse_dog_car_person_600x800.bgr";
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_GROUP_HANDLE hGroupHandle;
    HI_PROPOSAL_Param_S copParam[1] = {0};
    HI_RUNTIME_WK_INFO_S astWkInfo[1] = {0};
    memset(&copParam[0], 0, sizeof(copParam));
    memset(&astWkInfo[0], 0, sizeof(astWkInfo));

    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);

    printf("\n============================= rfcn net begin ================================\n");
    s32Ret = HI_SVPRT_RUNTIME_Init(CPU_TASK_AFFINITY, NULL);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_1, "HI_SVPRT_RUNTIME_Init failed!\n");

    s32Ret = SAMPLE_RUNTIME_LoadModelGroup_RFCN(pcRuntimeModelName, astWkInfo, copParam, &hGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_LoadModelGroup_RFCN failed!\n");

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Load]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = SAMPLE_RUNTIME_ForwardGroup_RFCN(pcSrcFile, hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "SAMPLE_RUNTIME_ForwardGroup_RFCN error\n");

    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Forward total]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = HI_SVPRT_RUNTIME_UnloadModelGroup(hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "HI_SVPRT_RUNTIME_UnloadModelGroup error\n");

FAIL_0:
    HI_SVPRT_RUNTIME_DeInit();
    releaseRfcnAndFrcnnCopParam(1, copParam);
    SAMPLE_FreeMem(&(astWkInfo[0].stWKMemory));

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Total]===== TIME SPEND: %ld ms =====\n", spend);
FAIL_1:
    SAMPLE_LOG_PRINT("SAMPLE_RFCN result %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", s32Ret);
    return s32Ret;
}
