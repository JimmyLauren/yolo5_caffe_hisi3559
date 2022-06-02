#include <time.h>
#include <stdio.h>
#ifdef ON_BOARD
#include <unistd.h>
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "mpi_ive.h"
#else
#include "back/hi_comm_svp.h"
#include "back/hi_nnie.h"
#include "back/mpi_nnie.h"
#include "sample_draw_rect.h"
#endif
#include "sample_resize_roi.h"
#include "sample_log.h"
#include "sample_memory_ops.h"
#include "sample_data_utils.h"
#include "sample_ive_image.h"

#define MAX_RESIZE_BATCH_NUM (300)

#ifdef ON_BOARD
HI_S32 blob2IveSrc(const HI_RUNTIME_BLOB_S* pstSrcBlob, const HI_S32 as32Coord[], IVE_SRC_IMAGE_S* pstIve)
{
    HI_U32 u32BlobOneChnSize = 0;
    HI_U32 u32IveOffset = 0;
    HI_S32 s32X1 = 0;
    HI_S32 s32X2 = 0;

    if (HI_RUNTIME_BLOB_TYPE_U8 == pstSrcBlob->enBlobType)
    {
        pstIve->enType = IVE_IMAGE_TYPE_U8C3_PLANAR;
    }
    else
    {
        printf("Not support blob type: %u \n", pstSrcBlob->enBlobType);
        return HI_FAILURE;
    }

    s32X1 = ALIGN16(as32Coord[0]);
    s32X2 = ALIGN16(as32Coord[2]);
    pstIve->u32Height = as32Coord[3] - as32Coord[1];
    pstIve->u32Width = s32X2 - s32X1;

    if (0 != pstIve->u32Height % 2)
    {
        pstIve->u32Height--;
    }

    if (0 != pstIve->u32Width % 2)
    {
        pstIve->u32Width--;
    }

    pstIve->au32Stride[0] = pstSrcBlob->u32Stride;
    pstIve->au32Stride[1] = pstSrcBlob->u32Stride;
    pstIve->au32Stride[2] = pstSrcBlob->u32Stride;

    u32BlobOneChnSize = pstSrcBlob->u32Stride * pstSrcBlob->unShape.stWhc.u32Height;
    u32IveOffset = as32Coord[1] * pstSrcBlob->u32Stride + s32X1;
    pstIve->au64PhyAddr[0] = pstSrcBlob->u64PhyAddr + u32IveOffset;
    pstIve->au64PhyAddr[1] = pstIve->au64PhyAddr[0] + u32BlobOneChnSize;
    pstIve->au64PhyAddr[2] = pstIve->au64PhyAddr[1] + u32BlobOneChnSize;

    pstIve->au64VirAddr[0] = pstSrcBlob->u64VirAddr + u32IveOffset;
    pstIve->au64VirAddr[1] = pstIve->au64VirAddr[0] + u32BlobOneChnSize;
    pstIve->au64VirAddr[2] = pstIve->au64VirAddr[1] + u32BlobOneChnSize;
    return HI_SUCCESS;
}

HI_BOOL IsSmallIveResizeRoi(const HI_S32 as32Coord[])
{
    HI_S32 s32Width = as32Coord[2] - as32Coord[0];
    HI_S32 s32Hight = as32Coord[3] - as32Coord[1];

    if ((s32Width < 32) && (s32Hight < 16))
    {
        return HI_TRUE;
    }

    return HI_FALSE;
}
HI_S32 checkIveResizeRoi(const HI_S32 as32Coord[])
{
    HI_S32 s32Width = as32Coord[2] - as32Coord[0];
    HI_S32 s32Hight = as32Coord[3] - as32Coord[1];

    if ((s32Width < 32) || (s32Width > 1920) || (s32Hight < 16) || (s32Hight > 1080))
    {
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

HI_S32 setIveDst(HI_RUNTIME_BLOB_S* pstDstBlob, HI_U16 u16Index, IVE_DST_IMAGE_S* pstIve)
{
    HI_U64 us64OneChnSize = 0;
    HI_U64 us64Size = 0;
    HI_U8 i = 0;

    if (HI_RUNTIME_BLOB_TYPE_U8 == pstDstBlob->enBlobType)
    {
        pstIve->enType = IVE_IMAGE_TYPE_U8C3_PLANAR;
    }
    else
    {
        printf("Not support blob type: %u \n", pstDstBlob->enBlobType);
        return HI_FAILURE;
    }

    pstIve->u32Height = pstDstBlob->unShape.stWhc.u32Height;
    pstIve->u32Width = pstDstBlob->unShape.stWhc.u32Width;

    if (0 != pstIve->u32Height % 2)
    {
        pstIve->u32Height--;
    }

    if (0 != pstIve->u32Width % 2)
    {
        pstIve->u32Width--;
    }

    if (0 != pstDstBlob->u32Stride % ALIGN_16)
    {
        printf("Stride error, not align 16 \n");
        return HI_FAILURE;
    }

    us64OneChnSize = (HI_U64)pstDstBlob->u32Stride * pstDstBlob->unShape.stWhc.u32Height;
    us64Size = us64OneChnSize * pstDstBlob->unShape.stWhc.u32Chn;

    for (i = 0; i < pstDstBlob->unShape.stWhc.u32Chn; i++)
    {
        pstIve->au32Stride[i] = pstDstBlob->u32Stride;
        pstIve->au64PhyAddr[i] = pstDstBlob->u64PhyAddr + u16Index * us64Size + i * us64OneChnSize;
        pstIve->au64VirAddr[i] = pstDstBlob->u64VirAddr + u16Index * us64Size + i * us64OneChnSize;
    }

    return HI_SUCCESS;
}

HI_S32 resizeROIByIVE(const HI_RUNTIME_BLOB_S* pstSrcBlob,
                                 const HI_S32 as32Coord[],
                                 HI_U32 u32CoordStride,
                                 HI_U16 u16Cnt,
                                 HI_RUNTIME_BLOB_S* pstDstBlob,
                                 HI_U16 u16DstOffSetCnt)
{
    IVE_HANDLE iveHandle = 0;
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32BlobSize = 0;
    HI_U16 i = 0;
    HI_BOOL bFinish = HI_FALSE;
    HI_RUNTIME_MEM_S stResizeMem;
#if PERFORMANCE_TEST
    long spend;
    struct timespec start, end;
#endif
    IVE_RESIZE_CTRL_S stResizeCtrl;
    HI_RUNTIME_MEM_S stMem;
    struct timespec request_time;
    request_time.tv_sec = 0;
    request_time.tv_nsec = 100 * 1000 * 1000;
    IVE_SRC_IMAGE_S *pstIveSrc = HI_NULL;
    IVE_DST_IMAGE_S *pstIveDst = HI_NULL;
    pstIveSrc = (IVE_SRC_IMAGE_S *)malloc(sizeof(IVE_SRC_IMAGE_S) * MAX_RESIZE_BATCH_NUM);
    SAMPLE_CHK_GOTO((HI_NULL == pstIveSrc), MALLOC_FREE, "malloc error!\n");
    pstIveDst = (IVE_DST_IMAGE_S *)malloc(sizeof(IVE_DST_IMAGE_S) * MAX_RESIZE_BATCH_NUM);
    SAMPLE_CHK_GOTO((HI_NULL == pstIveDst), MALLOC_FREE, "malloc error!\n");

    memset(&stResizeCtrl, 0x0, sizeof(IVE_RESIZE_CTRL_S));
    memset(pstIveSrc, 0x0, sizeof(IVE_SRC_IMAGE_S) * MAX_RESIZE_BATCH_NUM);
    memset(pstIveDst, 0x0, sizeof(IVE_DST_IMAGE_S) * MAX_RESIZE_BATCH_NUM);

    SAMPLE_CHK_GOTO((u16Cnt > MAX_RESIZE_BATCH_NUM), MALLOC_FREE, "Roi number[%u] exceed limit[%u]!", u16Cnt, MAX_RESIZE_BATCH_NUM);

    for (i = 0; i < u16Cnt; i++)
    {
        s32Ret = checkIveResizeRoi(&as32Coord[i * u32CoordStride]);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), MALLOC_FREE,
            "Check ive resize roi[x1=%d, y1=%d, x2=%d, y2=%d] failed!\n",
            as32Coord[i * u32CoordStride], as32Coord[i * u32CoordStride + 1],
            as32Coord[i * u32CoordStride + 2], as32Coord[i * u32CoordStride + 3]);

        s32Ret = blob2IveSrc(pstSrcBlob, &as32Coord[i * u32CoordStride], &pstIveSrc[i]);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), MALLOC_FREE, "blob2IveSrc fail:%#x", s32Ret);

        s32Ret = setIveDst(pstDstBlob, i + u16DstOffSetCnt, &pstIveDst[i]);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), MALLOC_FREE, "setIveDst fail:%#x", s32Ret);

    }

    stResizeCtrl.enMode = IVE_RESIZE_MODE_LINEAR;
    stResizeCtrl.u16Num = (HI_U16)u16Cnt;
    stResizeCtrl.stMem.u32Size = 49 * stResizeCtrl.u16Num;
    stResizeMem.u32Size = stResizeCtrl.stMem.u32Size;
    s32Ret = SAMPLE_AllocMem(&stResizeMem, HI_FALSE);
    stResizeCtrl.stMem.u64PhyAddr = stResizeMem.u64PhyAddr;
    stResizeCtrl.stMem.u64VirAddr = stResizeMem.u64VirAddr;
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), MMZ_FREE, "SAMPLE_Utils_AllocMem fail:%#x", s32Ret);

#if PERFORMANCE_TEST
    clock_gettime(0, &start);
#endif
    s32Ret = HI_MPI_IVE_Resize(&iveHandle, pstIveSrc, pstIveDst, &stResizeCtrl, HI_TRUE);
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), MMZ_FREE, "HI_MPI_IVE_Resize fail:%#x", s32Ret);

    s32Ret = HI_MPI_IVE_Query(iveHandle, &bFinish, HI_TRUE);
    while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
    {
        nanosleep(&request_time, HI_NULL);
        printf("HI_MPI_IVE_Query timeout, wait 100us\n");
        s32Ret = HI_MPI_IVE_Query(iveHandle, &bFinish, HI_TRUE);
    }
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, MMZ_FREE, "Error(%#x), HI_MPI_IVE_Query failed!\n", s32Ret);

    u32BlobSize = SAMPLE_DATA_GetBlobSize(pstDstBlob->u32Stride, pstDstBlob->u32Num, pstDstBlob->unShape.stWhc.u32Height, pstDstBlob->unShape.stWhc.u32Chn);
    stMem.u64PhyAddr = pstDstBlob->u64PhyAddr;
    stMem.u64VirAddr = pstDstBlob->u64VirAddr;
    stMem.u32Size = u32BlobSize;
    s32Ret = SAMPLE_FlushCache(&stMem);
    if (HI_SUCCESS != s32Ret)
    {
        printf("SAMPLE_FlushCache error after IVE\n");
    }

#if PERFORMANCE_TEST
    clock_gettime(0, &end);
    spend = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("===== IVE Resize TIME SPEND: %ld us =====\n", spend);
#endif

MMZ_FREE:
    SAMPLE_FREE(pstIveSrc);
    SAMPLE_FREE(pstIveDst);
    SAMPLE_FreeMem(&stResizeMem);
    return s32Ret;

MALLOC_FREE:
    SAMPLE_FREE(pstIveSrc);
    SAMPLE_FREE(pstIveDst);
    return s32Ret;
}
#endif

HI_S32 resizeROI(const HI_RUNTIME_BLOB_S* pstSrcBlob,
                            const HI_S32 as32Coord[],
                            HI_U32 u32CoordStride,
                            const HI_U16 u16RoiCnt,
                            HI_RUNTIME_BLOB_S* pstDstBlob,
                            const HI_U16 u16DstOffSetCnt)
{
#ifdef ON_BOARD
    HI_U16 i = 0;
    HI_BOOL bRet = HI_TRUE;

    for (i = 0; i < u16RoiCnt; i++)
    {
        bRet = IsSmallIveResizeRoi(&as32Coord[i * u32CoordStride]);

        if (HI_TRUE == bRet)
        {
            printf("Small resize roi[x1=%d, y1=%d, x2=%d, y2=%d]!\n",
                   as32Coord[i * u32CoordStride], as32Coord[i * u32CoordStride + 1],
                   as32Coord[i * u32CoordStride + 2], as32Coord[i * u32CoordStride + 3]);
            printf("OPENCV hasn't be linked, plz open it first!\n");
            return HI_FAILURE;
        }
    }

    return resizeROIByIVE(pstSrcBlob, as32Coord, u32CoordStride, u16RoiCnt, pstDstBlob, u16DstOffSetCnt);
#else
    return resizeROIByCV(pstSrcBlob, as32Coord, u32CoordStride, u16RoiCnt, pstDstBlob, u16DstOffSetCnt);
#endif
}

HI_S32 resizeBlob(const HI_RUNTIME_BLOB_S* pstSrcBlob,
                  HI_RUNTIME_BLOB_S* pstDstBlob)
{
    if ((3 != pstSrcBlob->unShape.stWhc.u32Chn) || (3 != pstDstBlob->unShape.stWhc.u32Chn))
    {
        SAMPLE_LOG_PRINT("Invalid input channel number, only support 3 for resizeBlob\n");
        return HI_FAILURE;
    }

#ifdef ON_BOARD
    return resizeByIVE(pstSrcBlob, pstDstBlob);
#else
    return resizeByCV(pstSrcBlob, pstDstBlob);
#endif
}


HI_S32 cropPadBlob(const HI_RUNTIME_BLOB_S* pstSrcBlob,
    const HI_RUNTIME_BLOB_S* pstBboxBlob,
    HI_RUNTIME_BLOB_S* pstDstBlob,
    TRACKER_LOCATION_S* pstLocation,
    HI_S32* ps32EdgeX,
    HI_S32* ps32EdgeY,
    HI_S32* ps32RegionW,
    HI_S32* ps32RegionH)
{
#ifdef ON_BOARD
    return cropPadBlobByIVE(pstSrcBlob, pstBboxBlob, pstDstBlob, pstLocation, ps32EdgeX, ps32EdgeY, ps32RegionW, ps32RegionH);
#else
    return cropPadBlobByCV(pstSrcBlob, pstBboxBlob, pstDstBlob, pstLocation, ps32EdgeX, ps32EdgeY, ps32RegionW, ps32RegionH);
#endif
}

HI_VOID drawImageRect(const HI_CHAR* pszPicPath, const HI_RUNTIME_BLOB_S* pstBlob,
                         HI_S32 as32Coord[], HI_U32 u32CoordCnt, HI_U32 u32CoordStride)
{
    HI_CHAR aszFileName[PATH_MAX] = {0};
#ifdef ON_BOARD
    snprintf(aszFileName, sizeof(aszFileName) - 1, "%s.ppm", pszPicPath);
    return drawImageRectByIVE(aszFileName, pstBlob, as32Coord, u32CoordCnt, u32CoordStride);
#else
    snprintf(aszFileName, sizeof(aszFileName) - 1, "%s.png", pszPicPath);
    return drawImageRectByCV(aszFileName, pstBlob, as32Coord, u32CoordCnt, u32CoordStride);
#endif
}
