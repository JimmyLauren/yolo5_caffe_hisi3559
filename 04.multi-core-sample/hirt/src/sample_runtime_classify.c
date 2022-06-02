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
#include "sample_runtime_classify.h"

#define MODEL_DIR RESOURCE_DIR
#define IMAGE_DIR RESOURCE_DIR
#define CONFIG_DIR "./"

HI_S32 SAMPLE_RUNTIME_LoadModelGroup_Alexnet(const HI_CHAR* pcModelFileAlex, HI_RUNTIME_WK_INFO_S* pstWkInfo,
        HI_RUNTIME_GROUP_HANDLE *phGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_GROUP_INFO_S stGroupInfo = {0};
    HI_CHAR acConfig[1024];
    memset(acConfig, 0, sizeof(acConfig));
    memset(&stGroupInfo, 0, sizeof(HI_RUNTIME_GROUP_INFO_S));
    strncpy(pstWkInfo[0].acModelName, "alexnet", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileAlex, &(pstWkInfo[0].stWKMemory));
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "SAMPLE_RUNTIME_LoadModelFile %s failed!\n", pcModelFileAlex);
    stGroupInfo.stWKsInfo.u32WKNum = 1;
    stGroupInfo.stWKsInfo.pstAttrs = &(pstWkInfo[0]);
    SAMPLE_RUNTIME_ReadConfig(CONFIG_DIR"alexnet.modelgroup", acConfig, 1024);
    acConfig[1023] = '\0';

    s32Ret = HI_SVPRT_RUNTIME_LoadModelGroup(acConfig, &stGroupInfo, phGroupHandle);
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "HI_SVPRT_RUNTIME_LoadModelGroup error\n");

    sample_debug("LoadGroup succ, group handle[%p]\n", pstGroup->pHiRTInternal);

    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetGroupSrc_FrcnnAlex(const HI_CHAR* pcSrcFile,
        HI_RUNTIME_BLOB_S* pstSrcAlexBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;

    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstSrcAlexBlobs[0], HI_RUNTIME_BLOB_TYPE_U8, 1, 227, 227, 3, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob alexnet failed!\n");

    s32Ret = SAMPLE_RUNTIME_ReadSrcFile(pcSrcFile, &(pstSrcAlexBlobs[0]));
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_ReadSrcFile failed!\n");

    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetGroupDst_FrcnnAlex(HI_RUNTIME_BLOB_S* pstDstAlexBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstAlexBlobs[0], HI_RUNTIME_BLOB_TYPE_VEC_S32, 1, 1, 1, 1000, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetBlob alexnet failed!\n");

    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_ForwardGroup_Alexnet(const HI_CHAR* pcSrcFile, HI_RUNTIME_GROUP_HANDLE hGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_GROUP_SRC_BLOB_ARRAY_S stGroupSrcBlob = {0};
    HI_RUNTIME_GROUP_DST_BLOB_ARRAY_S stGroupDstBlob = {0};
    HI_RUNTIME_GROUP_BLOB_S stInputBlob;
    HI_RUNTIME_GROUP_BLOB_S stOutputBlob;
    HI_RUNTIME_MEM_S stMem = {0};

    HI_RUNTIME_BLOB_S stSrc[1];
    HI_RUNTIME_BLOB_S stDst[1];
    memset(&stSrc[0], 0, sizeof(HI_RUNTIME_BLOB_S));
    memset(&stDst[0], 0, sizeof(HI_RUNTIME_BLOB_S));
    memset(&stInputBlob, 0, sizeof(HI_RUNTIME_GROUP_BLOB_S));
    memset(&stOutputBlob, 0, sizeof(HI_RUNTIME_GROUP_BLOB_S));

    stGroupSrcBlob.u32BlobNum = 1;
    stGroupSrcBlob.pstBlobs = &stInputBlob;
    stInputBlob.acOwnerName[0] = '\0';
    strncpy(stInputBlob.acBlobName, "data", MAX_NAME_LEN);
    stInputBlob.pstBlob = &stSrc[0];

    stGroupDstBlob.u32BlobNum = 1;
    stGroupDstBlob.pstBlobs = &stOutputBlob;
    //stOutputBlob.acOwnerName[0] = '\0';
    strncpy(stOutputBlob.acOwnerName, "alexnet", MAX_NAME_LEN);
    strncpy(stOutputBlob.acBlobName, "prob", MAX_NAME_LEN);
    stOutputBlob.pstBlob = &stDst[0];

    s32Ret = SAMPLE_RUNTIME_SetGroupSrc_FrcnnAlex(pcSrcFile, stSrc);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetGroupSrc_FrcnnAlex failed!\n");

    s32Ret = SAMPLE_RUNTIME_SetGroupDst_FrcnnAlex(stDst);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetGroupDst_FrcnnAlex failed!\n");
#if PERFORMANCE_TEST
    long spend;
    struct timespec start, end;
    clock_gettime(0, &start);
#endif

    s32Ret = HI_SVPRT_RUNTIME_ForwardGroupSync(hGroupHandle, &stGroupSrcBlob, &stGroupDstBlob, 0);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_SVPRT_RUNTIME_ForwardGroupSync failed!\n");
#if PERFORMANCE_TEST
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 * 1000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("\n\n[Forward]===== TIME SPEND: %ldms, %ldus =====\n\n", spend/1000, spend);
#endif

    SAMPLE_LOG_PRINT("Pic: %s\n", pcSrcFile);
    s32Ret = SAMPLE_RUNTIME_Cnn_TopN_Output(stGroupDstBlob.pstBlobs[0].pstBlob, 5);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "SAMPLE_RUNTIME_Cnn_TopN_Output error\n");
#if DEBUG
    printDebugData((HI_CHAR*)"alexnet", stDst[0].u64VirAddr, 10);
    s32Ret = HI_SUCCESS;
#endif

FAIL_0:

    stMem.u64PhyAddr = stSrc[0].u64PhyAddr;
    stMem.u64VirAddr = stSrc[0].u64VirAddr;
    SAMPLE_FreeMem(&stMem);

    stMem.u64PhyAddr = stDst[0].u64PhyAddr;
    stMem.u64VirAddr = stDst[0].u64VirAddr;
    SAMPLE_FreeMem(&stMem);
    return s32Ret;
}

HI_S32 SAMPLE_AlexNet()
{
#ifdef ON_BOARD
    const HI_CHAR* pcRuntimeModelName = MODEL_DIR"/runtime_alexnet_no_group_inst.wk";
#else
    const HI_CHAR* pcRuntimeModelName = MODEL_DIR"/runtime_alexnet_no_group_func.wk";
#endif
    const HI_CHAR* pcSrcFile = IMAGE_DIR"/ILSVRC2012_val_00000023_227x227.bgr";
    HI_S32 s32Ret = HI_FAILURE;

    HI_RUNTIME_GROUP_HANDLE hGroupHandle = HI_NULL;
    HI_RUNTIME_WK_INFO_S astWkInfo[1];

    memset(&astWkInfo[0], 0, sizeof(astWkInfo));

    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);

    printf("\n============================= alex net begin ================================\n");
    s32Ret = HI_SVPRT_RUNTIME_Init(CPU_TASK_AFFINITY, NULL);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_1, "HI_SVPRT_RUNTIME_Init failed!\n");

    s32Ret = SAMPLE_RUNTIME_LoadModelGroup_Alexnet(pcRuntimeModelName, astWkInfo, &hGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_LoadModelGroup_Alexnet failed!\n");

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Load]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = SAMPLE_RUNTIME_ForwardGroup_Alexnet(pcSrcFile, hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "SAMPLE_RUNTIME_ForwardGroup_Alexnet error\n");

    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Forward total]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = HI_SVPRT_RUNTIME_UnloadModelGroup(hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "HI_SVPRT_RUNTIME_UnloadModelGroup error\n");

FAIL_0:
    HI_SVPRT_RUNTIME_DeInit();
    SAMPLE_FreeMem(&(astWkInfo[0].stWKMemory));

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Total]===== TIME SPEND: %ld ms =====\n", spend);
FAIL_1:
    SAMPLE_LOG_PRINT("SAMPLE_AlexNet result %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", s32Ret);
    return s32Ret;
}
