#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hi_runtime_api.h"
#include "sample_memory_ops.h"
#include "sample_log.h"
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
#include "sample_runtime_detection_ssd.h"

#define MODEL_DIR RESOURCE_DIR
#define IMAGE_DIR RESOURCE_DIR
#define CONFIG_DIR "./"

HI_S32 SAMPLE_RUNTIME_LoadModelGroup_SSD(
        const HI_CHAR* pcModelFileSSD,
        HI_RUNTIME_WK_INFO_S *pstWKInfo,
        HI_RUNTIME_GROUP_HANDLE* phGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_GROUP_INFO_S stGroupInfo;
    HI_CHAR acConfig[1024];

    memset(acConfig, 0, sizeof(acConfig));
    memset(&stGroupInfo, 0, sizeof(HI_RUNTIME_GROUP_INFO_S));

    stGroupInfo.stWKsInfo.u32WKNum = 1;
    strncpy(pstWKInfo->acModelName, "ssd", MAX_NAME_LEN);
    s32Ret = SAMPLE_RUNTIME_LoadModelFile(pcModelFileSSD, &pstWKInfo->stWKMemory);
    SAMPLE_CHK_RETURN(HI_SUCCESS != s32Ret, HI_FAILURE, "LoadFile fail");
    stGroupInfo.stWKsInfo.pstAttrs = pstWKInfo;

    SAMPLE_RUNTIME_ReadConfig(CONFIG_DIR"ssd.modelgroup", acConfig, 1024);
    acConfig[1023] = '\0';

    s32Ret = HI_SVPRT_RUNTIME_LoadModelGroup(acConfig, &stGroupInfo, phGroupHandle);
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0,
                                   "HI_SVPRT_RUNTIME_LoadModelGroup error\n");

    sample_debug("LoadGroup succ, group handle[%p]\n", pstGroup->pHiRTInternal);
    return HI_SUCCESS;

FAIL_0:
    SAMPLE_FreeMem(&pstWKInfo->stWKMemory);
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetGroupSrc_SSD(const HI_CHAR* pcSrcFile, HI_RUNTIME_BLOB_S* pstSrcSSDBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_RUNTIME_MEM_S stMem;

    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstSrcSSDBlobs[0], HI_RUNTIME_BLOB_TYPE_U8, 1, 300, 300, 3, ALIGN_16);
    SAMPLE_CHK_RETURN(HI_SUCCESS != s32Ret, HI_FAILURE, "SAMPLE_RUNTIME_SetBlob ssd failed!\n");

    s32Ret = SAMPLE_RUNTIME_ReadSrcFile(pcSrcFile, &(pstSrcSSDBlobs[0]));
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_ReadSrcFile failed!\n");

    return HI_SUCCESS;
FAIL_0:
    stMem.u64PhyAddr = pstSrcSSDBlobs[0].u64PhyAddr;
    stMem.u64VirAddr = pstSrcSSDBlobs[0].u64VirAddr;
    SAMPLE_FreeMem(&stMem);
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetGroupDst_SSD(HI_RUNTIME_BLOB_S* pstDstSSDBlobs)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32DstIndex = 0;

    //conv4_3_norm_mbox_loc_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 16, 38, 38, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv4_3_norm_mbox_loc_perm failed!\n");
    //conv4_3_norm_mbox_conf_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 84, 38, 38, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv4_3_norm_mbox_conf_perm failed!\n");
    //fc7_mbox_loc_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 24, 19, 19, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst fc7_mbox_loc_perm failed!\n");
    //fc7_mbox_conf_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 126, 19, 19, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst fc7_mbox_conf_perm failed!\n");
    //conv6_2_mbox_loc_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 24, 10, 10, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv6_2_mbox_loc_perm failed!\n");
    //conv6_2_mbox_conf_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 126, 10, 10, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv6_2_mbox_conf_perm failed!\n");
    //conv7_2_mbox_loc_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 24, 5, 5, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv7_2_mbox_loc_perm failed!\n");
    //conv7_2_mbox_conf_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 126, 5, 5, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv7_2_mbox_conf_perm failed!\n");
    //conv8_2_mbox_loc_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 16, 3, 3, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv8_2_mbox_loc_perm failed!\n");
    //conv8_2_mbox_conf_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 84, 3, 3, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv8_2_mbox_conf_perm failed!\n");
    //conv9_2_mbox_loc_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 16, 1, 1, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv9_2_mbox_loc_perm failed!\n");
    //conv9_2_mbox_conf_perm
    s32Ret = SAMPLE_RUNTIME_SetBlob((HI_RUNTIME_BLOB_S*)&pstDstSSDBlobs[u32DstIndex++], HI_RUNTIME_BLOB_TYPE_S32, 1, 84, 1, 1, ALIGN_16);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_SetBlob ssd dst conv9_2_mbox_conf_perm failed!\n");
    return HI_SUCCESS;
FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_ForwardGroup_SSD(const HI_CHAR* pcSrcFile, HI_RUNTIME_GROUP_HANDLE hGroupHandle)
{
    HI_S32 s32Ret = HI_FAILURE;
#if DEBUG
    HI_CHAR* pcOutName = HI_NULL;
#endif
    HI_S32 as32ResultROI[200 * SVP_WK_PROPOSAL_WIDTH] = { 0 };
    HI_U32 u32ResultROICnt = 0;
    HI_RUNTIME_MEM_S stMem;
    HI_RUNTIME_GROUP_SRC_BLOB_ARRAY_S stSrcGroupBlobArray;
    HI_RUNTIME_GROUP_DST_BLOB_ARRAY_S stDstGroupBlobArray;

    HI_RUNTIME_GROUP_BLOB_S astSrcGroupBlobs[1];
    HI_RUNTIME_GROUP_BLOB_S astDstGroupBlobs[12];
    HI_RUNTIME_BLOB_S stSrc[1];
    HI_RUNTIME_BLOB_S stDst[12];

    memset(&astSrcGroupBlobs[0], 0, sizeof(astSrcGroupBlobs));
    memset(&astDstGroupBlobs[0], 0, sizeof(astDstGroupBlobs));

    stSrcGroupBlobArray.u32BlobNum = 1;
    stDstGroupBlobArray.u32BlobNum = 12;
    stSrcGroupBlobArray.pstBlobs = astSrcGroupBlobs;
    stDstGroupBlobArray.pstBlobs = astDstGroupBlobs;

    strncpy(astSrcGroupBlobs[0].acOwnerName, "", MAX_NAME_LEN);
    strncpy(astSrcGroupBlobs[0].acBlobName, "data", MAX_NAME_LEN);
    astSrcGroupBlobs[0].pstBlob = stSrc;

    strncpy(astDstGroupBlobs[0].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[0].acBlobName, "conv4_3_norm_mbox_loc_perm", MAX_NAME_LEN);
    astDstGroupBlobs[0].pstBlob = &stDst[0];
    strncpy(astDstGroupBlobs[1].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[1].acBlobName, "conv4_3_norm_mbox_conf_perm", MAX_NAME_LEN);
    astDstGroupBlobs[1].pstBlob = &stDst[1];

    strncpy(astDstGroupBlobs[2].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[2].acBlobName, "fc7_mbox_loc_perm", MAX_NAME_LEN);
    astDstGroupBlobs[2].pstBlob = &stDst[2];
    strncpy(astDstGroupBlobs[3].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[3].acBlobName, "fc7_mbox_conf_perm", MAX_NAME_LEN);
    astDstGroupBlobs[3].pstBlob = &stDst[3];

    strncpy(astDstGroupBlobs[4].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[4].acBlobName, "conv6_2_mbox_loc_perm", MAX_NAME_LEN);
    astDstGroupBlobs[4].pstBlob = &stDst[4];
    strncpy(astDstGroupBlobs[5].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[5].acBlobName, "conv6_2_mbox_conf_perm", MAX_NAME_LEN);
    astDstGroupBlobs[5].pstBlob = &stDst[5];

    strncpy(astDstGroupBlobs[6].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[6].acBlobName, "conv7_2_mbox_loc_perm", MAX_NAME_LEN);
    astDstGroupBlobs[6].pstBlob = &stDst[6];
    strncpy(astDstGroupBlobs[7].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[7].acBlobName, "conv7_2_mbox_conf_perm", MAX_NAME_LEN);
    astDstGroupBlobs[7].pstBlob = &stDst[7];

    strncpy(astDstGroupBlobs[8].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[8].acBlobName, "conv8_2_mbox_loc_perm", MAX_NAME_LEN);
    astDstGroupBlobs[8].pstBlob = &stDst[8];
    strncpy(astDstGroupBlobs[9].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[9].acBlobName, "conv8_2_mbox_conf_perm", MAX_NAME_LEN);
    astDstGroupBlobs[9].pstBlob = &stDst[9];

    strncpy(astDstGroupBlobs[10].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[10].acBlobName, "conv9_2_mbox_loc_perm", MAX_NAME_LEN);
    astDstGroupBlobs[10].pstBlob = &stDst[10];
    strncpy(astDstGroupBlobs[11].acOwnerName, "ssd", MAX_NAME_LEN);
    strncpy(astDstGroupBlobs[11].acBlobName, "conv9_2_mbox_conf_perm", MAX_NAME_LEN);
    astDstGroupBlobs[11].pstBlob = &stDst[11];

    memset(stSrc, 0, sizeof(stSrc));
    memset(stDst, 0, sizeof(stDst));

    s32Ret = SAMPLE_RUNTIME_SetGroupSrc_SSD(pcSrcFile, stSrc);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetGroupSrc_FrcnnAlex failed!\n");

    s32Ret = SAMPLE_RUNTIME_SetGroupDst_SSD(stDst);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_SetGroupDst_FrcnnAlex failed!\n");

    s32Ret = HI_SVPRT_RUNTIME_ForwardGroupSync(hGroupHandle, &stSrcGroupBlobArray, &stDstGroupBlobArray, 0);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_SVPRT_RUNTIME_ForwardGroupSync failed!\n");

    SAMPLE_LOG_PRINT("Pic: %s\n", pcSrcFile);
    SAMPLE_Ssd_GetResult(stSrc, stDst, as32ResultROI, &u32ResultROICnt);
    SAMPLE_LOG_PRINT("roi cnt: %d\n", u32ResultROICnt);

    drawImageRect("ssd", &stSrc[0], as32ResultROI, u32ResultROICnt, 6);

#if DEBUG
    for (int i = 0; i < 12; i++)
    {
        switch(i) {
            case 0:
                pcOutName = (HI_CHAR*)"conv4_3_norm_mbox_loc_perm";
                break;
            case 1:
                pcOutName = (HI_CHAR*)"conv4_3_norm_mbox_conf_perm";
                break;
            case 2:
                pcOutName = (HI_CHAR*)"fc7_mbox_loc_perm";
                break;
            case 3:
                pcOutName = (HI_CHAR*)"fc7_mbox_conf_perm";
                break;
            case 4:
                pcOutName = (HI_CHAR*)"conv6_2_mbox_loc_perm";
                break;
            case 5:
                pcOutName = (HI_CHAR*)"conv6_2_mbox_conf_perm";
                break;
            case 6:
                pcOutName = (HI_CHAR*)"conv7_2_mbox_loc_perm";
                break;
            case 7:
                pcOutName = (HI_CHAR*)"conv7_2_mbox_conf_perm";
                break;
            case 8:
                pcOutName = (HI_CHAR*)"conv8_2_mbox_loc_perm";
                break;
            case 9:
                pcOutName = (HI_CHAR*)"conv8_2_mbox_conf_perm";
                break;
            case 10:
                pcOutName = (HI_CHAR*)"conv9_2_mbox_loc_perm";
                break;
            case 11:
                pcOutName = (HI_CHAR*)"conv9_2_mbox_conf_perm";
                break;
            default:
                break;
        }
        printDebugData(pcOutName, stDst[i].u64VirAddr, 10);
    }

#endif

FAIL_0:

    stMem.u64PhyAddr = stSrc[0].u64PhyAddr;
    stMem.u64VirAddr = stSrc[0].u64VirAddr;
    SAMPLE_FreeMem(&stMem);

    for (HI_U32 i = 0; i < 12; ++i) {
        stMem.u64PhyAddr = stDst[i].u64PhyAddr;
        stMem.u64VirAddr = stDst[i].u64VirAddr;
        SAMPLE_FreeMem(&stMem);
    }
    return s32Ret;
}

HI_S32 SAMPLE_SSD()
{
#ifdef ON_BOARD
    const HI_CHAR* pcRuntimeModelName = MODEL_DIR"/runtime_ssd_inst.wk";
#else
    const HI_CHAR* pcRuntimeModelName = MODEL_DIR"/runtime_ssd_func.wk";
#endif
    const HI_CHAR* pcSrcFile = IMAGE_DIR"/dog_bike_car_300x300.bgr";
    HI_RUNTIME_WK_INFO_S stWKInfo;
    HI_S32 s32Ret = HI_FAILURE;

    HI_RUNTIME_GROUP_HANDLE hGroupHandle;

    memset(&stWKInfo, 0, sizeof(HI_RUNTIME_WK_INFO_S));

    long spend;
    struct timespec start, next, end;
    clock_gettime(0, &start);

    printf("\n============================= ssd net begin ================================\n");
    s32Ret = HI_SVPRT_RUNTIME_Init(CPU_TASK_AFFINITY, NULL);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_1, "HI_SVPRT_RUNTIME_Init failed!\n");

    s32Ret = SAMPLE_RUNTIME_LoadModelGroup_SSD(pcRuntimeModelName, &stWKInfo, &hGroupHandle);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_RUNTIME_LoadModelGroup_SSD failed!\n");

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Load]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = SAMPLE_RUNTIME_ForwardGroup_SSD(pcSrcFile, hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "SAMPLE_RUNTIME_ForwardGroup_SSD error\n");

    clock_gettime(0, &next);
    spend = (next.tv_sec - end.tv_sec) * 1000 + (next.tv_nsec - end.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Forward total]===== TIME SPEND: %ld ms =====\n", spend);

    s32Ret = HI_SVPRT_RUNTIME_UnloadModelGroup(hGroupHandle);
    SAMPLE_CHK_PRINTF((HI_SUCCESS != s32Ret), "HI_SVPRT_RUNTIME_UnloadModelGroup error\n");

FAIL_0:
    SAMPLE_FreeMem(&stWKInfo.stWKMemory);
    (HI_VOID)HI_SVPRT_RUNTIME_DeInit();

    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    SAMPLE_LOG_PRINT("\n[Total]===== TIME SPEND: %ld ms =====\n", spend);
FAIL_1:
    SAMPLE_LOG_PRINT("SAMPLE_SSD result %d !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", s32Ret);
    return s32Ret;
}
