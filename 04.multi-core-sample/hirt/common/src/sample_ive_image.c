#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#ifdef ON_BOARD
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include "math.h"
#include "mpi_ive.h"
#include "mpi_sys.h"
#include "hi_comm_ive.h"
#include "sample_log.h"
#include "sample_memory_ops.h"
#include "sample_save_blob.h"
#include "sample_resize_roi.h"
#include "sample_data_utils.h"

HI_S32 ive_copy_singal_small(IVE_DATA_S *pstSrc, IVE_DATA_S *pstDst)
{
    HI_U8 *pu8Src = (HI_U8 *)((HI_UL)pstSrc->u64VirAddr);
    HI_U8 *pu8Dst = (HI_U8 *)((HI_UL)pstDst->u64VirAddr);
    for (HI_U32 i = 0; i < pstSrc->u32Height; i++)
    {
        for (HI_U32 j = 0; j < pstSrc->u32Width; j++)
        {
            pu8Dst[i * pstDst->u32Stride + j] = pu8Src[i * pstSrc->u32Stride + j];
        }
    }
    return HI_SUCCESS;
}

HI_S32 iveDmaCopy(IVE_HANDLE *pIveHandle, IVE_DATA_S *pstSrc,
    IVE_DST_DATA_S *pstDst, IVE_DMA_CTRL_S *pstDmaCtrl,HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_BOOL bFinish = HI_FALSE;
    struct timespec request_time;

    request_time.tv_sec = 0;
    request_time.tv_nsec = 100 * 1000 * 1000;

    if ((pstSrc->u32Height < 32) || (pstSrc->u32Width < 32))
    {
        return ive_copy_singal_small(pstSrc, pstDst);
    }
    else
    {
        s32Ret = HI_MPI_IVE_DMA(pIveHandle, pstSrc, pstDst, pstDmaCtrl, bInstant);
        if ((HI_TRUE == bInstant) && (HI_SUCCESS == s32Ret))
        {
            s32Ret = HI_MPI_IVE_Query(*pIveHandle, &bFinish, HI_TRUE);
            while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
            {
                nanosleep(&request_time, HI_NULL);
                printf("HI_MPI_IVE_Query timeout, wait 100us\n");
                s32Ret = HI_MPI_IVE_Query(*pIveHandle, &bFinish, HI_TRUE);
            }
        }
        return s32Ret;
    }
}

static HI_U16 iveCalcStride(IVE_IMAGE_TYPE_E enType, HI_U32 u32Width, HI_U8 u8Align)
{
    HI_U32 u32Size = 0;

    switch (enType)
    {
        case IVE_IMAGE_TYPE_U8C1:
        case IVE_IMAGE_TYPE_S8C1:
        case IVE_IMAGE_TYPE_S8C2_PACKAGE:
        case IVE_IMAGE_TYPE_S8C2_PLANAR:
        case IVE_IMAGE_TYPE_U8C3_PACKAGE:
        case IVE_IMAGE_TYPE_U8C3_PLANAR:
        {
            u32Size = sizeof(HI_U8);
        }
        break;
        case IVE_IMAGE_TYPE_S16C1:
        case IVE_IMAGE_TYPE_U16C1:
        {

            u32Size = sizeof(HI_U16);
        }
        break;
        case IVE_IMAGE_TYPE_S32C1:
        case IVE_IMAGE_TYPE_U32C1:
        {
            u32Size = sizeof(HI_U32);
        }
        break;
        case IVE_IMAGE_TYPE_S64C1:
        case IVE_IMAGE_TYPE_U64C1:
        {
            u32Size = sizeof(HI_U64);
        }
        break;
        default:
            break;

    }

    if (ALIGN_16 == u8Align)
    {
        return ALIGN16(u32Width * u32Size);
    }
    else
    {
        return ALIGN32(u32Width * u32Size);
    }
}

static HI_VOID setIveImage(IVE_IMAGE_S* pstImg, HI_U8 u8C1, HI_U8 u8C2, HI_U8 u8C3)
{
    HI_U32 u32OneCSize = pstImg->au32Stride[0] * pstImg->u32Height;
    memset((HI_U8*)((HI_UL)pstImg->au64VirAddr[0]), (HI_S32)u8C1, u32OneCSize);
    memset((HI_U8*)((HI_UL)pstImg->au64VirAddr[1]), (HI_S32)u8C2, u32OneCSize);
    memset((HI_U8*)((HI_UL)pstImg->au64VirAddr[2]), (HI_S32)u8C3, u32OneCSize);
}

static HI_S32 createImageByIVE(IVE_IMAGE_S* pstImg, IVE_IMAGE_TYPE_E enType, HI_U32 u32Width, HI_U32 u32Height)
{
    HI_U32 u32Size = 0;
    HI_U32 u32OneCSize = 0;
    HI_S32 s32Ret;
    if (NULL == pstImg)
    {
        SAMPLE_LOG_PRINT("pstImg is null\n");
        return HI_FAILURE;
    }

    pstImg->enType = enType;
    pstImg->u32Width = u32Width;
    pstImg->u32Height = u32Height;
    pstImg->au32Stride[0] = iveCalcStride(pstImg->enType, pstImg->u32Width, ALIGN_16);

    switch (enType)
    {
        case IVE_IMAGE_TYPE_U8C1:
        case IVE_IMAGE_TYPE_S8C1:
        {
            u32Size = pstImg->au32Stride[0] * pstImg->u32Height;
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
        }
        break;
        case IVE_IMAGE_TYPE_YUV420SP:
        {
            u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 3 / 2;
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
            pstImg->au32Stride[1] = pstImg->au32Stride[0];
            pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * (HI_U64)pstImg->u32Height;
            pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * (HI_U64)pstImg->u32Height;

        }
        break;
        case IVE_IMAGE_TYPE_YUV422SP:
        {
            u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 2;
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
            pstImg->au32Stride[1] = pstImg->au32Stride[0];
            pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + pstImg->au32Stride[0] * (HI_U64)pstImg->u32Height;
            pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + pstImg->au32Stride[0] * (HI_U64)pstImg->u32Height;

        }
        break;
        case IVE_IMAGE_TYPE_YUV420P:
            break;
        case IVE_IMAGE_TYPE_YUV422P:
            break;
        case IVE_IMAGE_TYPE_S8C2_PACKAGE:
            break;
        case IVE_IMAGE_TYPE_S8C2_PLANAR:
            break;
        case IVE_IMAGE_TYPE_S16C1:
        case IVE_IMAGE_TYPE_U16C1:
        {

            u32Size = pstImg->au32Stride[0] * pstImg->u32Height * sizeof(HI_U16);
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
        }
        break;
        case IVE_IMAGE_TYPE_U8C3_PACKAGE:
        {
            u32Size = pstImg->au32Stride[0] * pstImg->u32Height * 3;
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
            pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + 1;
            pstImg->au64VirAddr[2] = pstImg->au64VirAddr[1] + 1;
            pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + 1;
            pstImg->au64PhyAddr[2] = pstImg->au64PhyAddr[1] + 1;
            pstImg->au32Stride[1] = pstImg->au32Stride[0];
            pstImg->au32Stride[2] = pstImg->au32Stride[0];
        }
        break;
        case IVE_IMAGE_TYPE_U8C3_PLANAR:
        {
            // every channel has the same stride
            u32OneCSize = pstImg->au32Stride[0] * pstImg->u32Height;
            u32Size = u32OneCSize * 3;
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
            pstImg->au64VirAddr[1] = pstImg->au64VirAddr[0] + u32OneCSize;
            pstImg->au64VirAddr[2] = pstImg->au64VirAddr[1] + u32OneCSize;
            pstImg->au64PhyAddr[1] = pstImg->au64PhyAddr[0] + u32OneCSize;
            pstImg->au64PhyAddr[2] = pstImg->au64PhyAddr[1] + u32OneCSize;
            pstImg->au32Stride[1] = pstImg->au32Stride[0];
            pstImg->au32Stride[2] = pstImg->au32Stride[0];
        }
        break;
        case IVE_IMAGE_TYPE_S32C1:
        case IVE_IMAGE_TYPE_U32C1:
        {
            u32Size = pstImg->au32Stride[0] * pstImg->u32Height * sizeof(HI_U32);
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
        }
        break;
        case IVE_IMAGE_TYPE_S64C1:
        case IVE_IMAGE_TYPE_U64C1:
        {

            u32Size = pstImg->au32Stride[0] * pstImg->u32Height * sizeof(HI_U64);
            s32Ret = HI_MPI_SYS_MmzAlloc(&pstImg->au64PhyAddr[0], (HI_VOID**)&pstImg->au64VirAddr[0], HI_NULL, HI_NULL, u32Size);
            if (s32Ret != HI_SUCCESS)
            {
                SAMPLE_LOG_PRINT("Mmz Alloc fail,Error(%#x)\n", s32Ret);
                return s32Ret;
            }
        }
        break;
        default:
            break;

    }

    return HI_SUCCESS;
}

static HI_VOID freeIveImage(IVE_IMAGE_S* pstImage)
{
    if ((0 != pstImage->au64PhyAddr[0]) && (0 != pstImage->au64VirAddr[0]))
    {
        HI_MPI_SYS_MmzFree(pstImage->au64PhyAddr[0], (HI_VOID*)((HI_UL)pstImage->au64VirAddr[0]));
        pstImage->au64PhyAddr[0] = 0;
        pstImage->au64VirAddr[0] = 0;
    }
    memset(pstImage, 0, sizeof(*pstImage));
}

static HI_S32 copyImageByIVE(IVE_IMAGE_S* pstSrcImage, IVE_IMAGE_S* pstDstImage, IVE_RECT_U16_S* pstDstRect)
{
    HI_S32 s32Ret = HI_FAILURE;
    IVE_HANDLE hIveHandle;
    IVE_DATA_S stSrc;
    IVE_DATA_S stDst;
    IVE_DMA_CTRL_S stDmaCtrl = {IVE_DMA_MODE_DIRECT_COPY, 0, 0, 0, 0};
    HI_U64 u64Offset = pstDstImage->au32Stride[0] * (HI_U64)pstDstRect->u16Y + (HI_U64)pstDstRect->u16X * sizeof(HI_U8);

    if (IVE_IMAGE_TYPE_U8C3_PLANAR != pstDstImage->enType)
    {
        SAMPLE_LOG_PRINT("Invalid image type[%u]\n", pstDstImage->enType);
        return HI_FAILURE;
    }

    stSrc.u32Height = pstDstRect->u16Height;
    stSrc.u32Width = pstDstRect->u16Width;

    stDst.u32Height = pstDstRect->u16Height;
    stDst.u32Width = pstDstRect->u16Width;

    stSrc.u32Stride = pstSrcImage->au32Stride[0];
    stSrc.u64PhyAddr = pstSrcImage->au64PhyAddr[0];
    stSrc.u64VirAddr = pstSrcImage->au64VirAddr[0];
    stDst.u32Stride = pstDstImage->au32Stride[0];
    stDst.u64PhyAddr = pstDstImage->au64PhyAddr[0] + u64Offset;
    stDst.u64VirAddr = pstDstImage->au64VirAddr[0] + u64Offset;
    s32Ret = iveDmaCopy(&hIveHandle, &stSrc, &stDst, &stDmaCtrl, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "iveDmaCopy 0 fail!\n");

    stSrc.u32Stride = pstSrcImage->au32Stride[1];
    stSrc.u64PhyAddr = pstSrcImage->au64PhyAddr[1];
    stSrc.u64VirAddr = pstSrcImage->au64VirAddr[1];
    stDst.u32Stride = pstDstImage->au32Stride[1];
    stDst.u64PhyAddr = pstDstImage->au64PhyAddr[1] + u64Offset;
    stDst.u64VirAddr = pstDstImage->au64VirAddr[1] + u64Offset;
    s32Ret = iveDmaCopy(&hIveHandle, &stSrc, &stDst, &stDmaCtrl, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "iveDmaCopy 1 fail!\n");

    stSrc.u32Stride = pstSrcImage->au32Stride[2];
    stSrc.u64PhyAddr = pstSrcImage->au64PhyAddr[2];
    stSrc.u64VirAddr = pstSrcImage->au64VirAddr[2];
    stDst.u32Stride = pstDstImage->au32Stride[2];
    stDst.u64PhyAddr = pstDstImage->au64PhyAddr[2] + u64Offset;
    stDst.u64VirAddr = pstDstImage->au64VirAddr[2] + u64Offset;
    s32Ret = iveDmaCopy(&hIveHandle, &stSrc, &stDst, &stDmaCtrl, HI_TRUE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "iveDmaCopy 2 fail!\n");

FAIL_0:
    return s32Ret;
}

static HI_S32 copyIveImage2Blob(const IVE_IMAGE_S* pstSrcImage, HI_RUNTIME_BLOB_S* pstDstBlobs, HI_U16 u16Index)
{
    HI_U32 u32DstWidth = pstDstBlobs->unShape.stWhc.u32Width;
    HI_U32 u32DstHeight = pstDstBlobs->unShape.stWhc.u32Height;
    HI_U32 u32DstStride = pstDstBlobs->u32Stride;
    HI_U8* pu8Dst = (HI_U8*)((HI_UL)(pstDstBlobs->u64VirAddr + u16Index * u32DstHeight * u32DstStride * pstDstBlobs->unShape.stWhc.u32Chn));
    HI_U8* pu8DstR = pu8Dst;
    HI_U8* pu8DstG = pu8DstR + u32DstStride * u32DstHeight;
    HI_U8* pu8DstB = pu8DstG + u32DstStride * u32DstHeight;

    if ((u32DstStride != pstSrcImage->au32Stride[0])
        || (u32DstHeight != pstSrcImage->u32Height)
        || (u32DstWidth != pstSrcImage->u32Width))
    {
        return HI_FAILURE;
    }

    HI_U32 u32OneCSize = u32DstStride * u32DstHeight;

    memcpy(pu8DstR, (HI_U8*)(HI_UL)pstSrcImage->au64VirAddr[0], u32OneCSize);
    memcpy(pu8DstG, (HI_U8*)(HI_UL)pstSrcImage->au64VirAddr[1], u32OneCSize);
    memcpy(pu8DstB, (HI_U8*)(HI_UL)pstSrcImage->au64VirAddr[2], u32OneCSize);

    return HI_SUCCESS;
}

static HI_S32 copyBlob2IveImage(const HI_RUNTIME_BLOB_S* pstSrcBlob, HI_U16 u16Index, IVE_IMAGE_S* pstImage)
{
    HI_U32 u32SrcWidth = pstSrcBlob->unShape.stWhc.u32Width;
    HI_U32 u32SrcHeight = pstSrcBlob->unShape.stWhc.u32Height;
    HI_U32 u32SrcStride = pstSrcBlob->u32Stride;
    HI_U8* pu8Src = (HI_U8*)(HI_UL)(pstSrcBlob->u64VirAddr + u16Index * u32SrcHeight * u32SrcStride * pstSrcBlob->unShape.stWhc.u32Chn);
    HI_U8* pu8SrcR = pu8Src;
    HI_U8* pu8SrcG = pu8SrcR + u32SrcStride * u32SrcHeight;
    HI_U8* pu8SrcB = pu8SrcG + u32SrcStride * u32SrcHeight;

    if ((u32SrcStride != pstImage->au32Stride[0])
        || (u32SrcHeight != pstImage->u32Height)
        || (u32SrcWidth != pstImage->u32Width))
    {
        return HI_FAILURE;
    }

    HI_U32 u32OneCSize = u32SrcStride * u32SrcHeight;

    memcpy((HI_U8*)((HI_UL)pstImage->au64VirAddr[0]), pu8SrcR, u32OneCSize);
    memcpy((HI_U8*)((HI_UL)pstImage->au64VirAddr[1]), pu8SrcG, u32OneCSize);
    memcpy((HI_U8*)((HI_UL)pstImage->au64VirAddr[2]), pu8SrcB, u32OneCSize);

    return HI_SUCCESS;
}


static HI_S32 createRectImageByIVE(const IVE_IMAGE_S* pstImage, IVE_RECT_U16_S* pstRect, IVE_IMAGE_S* pstDstImage)
{
    HI_S32 s32Ret = HI_FAILURE;
    IVE_HANDLE hIveHandle;
    IVE_DATA_S stSrc;
    IVE_DATA_S stDst;
    IVE_DMA_CTRL_S stDmaCtrl = {IVE_DMA_MODE_DIRECT_COPY, 0, 0, 0, 0};
    HI_U64 u64Offset = pstImage->au32Stride[0] * (HI_U64)pstRect->u16Y + (HI_U64)pstRect->u16X * sizeof(HI_U8);

    if (IVE_IMAGE_TYPE_U8C3_PLANAR != pstImage->enType)
    {
        SAMPLE_LOG_PRINT("Invalid image type[%u]\n", pstImage->enType);
        return HI_FAILURE;
    }

    s32Ret = createImageByIVE(pstDstImage, IVE_IMAGE_TYPE_U8C3_PLANAR, (HI_U32)pstRect->u16Width, (HI_U32)pstRect->u16Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail!\n");

    stSrc.u32Height = pstRect->u16Height;
    stSrc.u32Width = pstRect->u16Width;

    stDst.u32Height = pstRect->u16Height;
    stDst.u32Width = pstRect->u16Width;

    stSrc.u32Stride = pstImage->au32Stride[0];
    stSrc.u64PhyAddr = pstImage->au64PhyAddr[0] + u64Offset;
    stSrc.u64VirAddr = pstImage->au64VirAddr[0] + u64Offset;
    stDst.u32Stride = pstDstImage->au32Stride[0];
    stDst.u64PhyAddr = pstDstImage->au64PhyAddr[0];
    stDst.u64VirAddr = pstDstImage->au64VirAddr[0];
    s32Ret = iveDmaCopy(&hIveHandle, &stSrc, &stDst, &stDmaCtrl, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "iveDmaCopy 0 fail!\n");

    stSrc.u32Stride = pstImage->au32Stride[1];
    stSrc.u64PhyAddr = pstImage->au64PhyAddr[1] + u64Offset;
    stSrc.u64VirAddr = pstImage->au64VirAddr[1] + u64Offset;
    stDst.u32Stride = pstDstImage->au32Stride[1];
    stDst.u64PhyAddr = pstDstImage->au64PhyAddr[1];
    stDst.u64VirAddr = pstDstImage->au64VirAddr[1];
    s32Ret = iveDmaCopy(&hIveHandle, &stSrc, &stDst, &stDmaCtrl, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "iveDmaCopy 1 fail!\n");

    stSrc.u32Stride = pstImage->au32Stride[2];
    stSrc.u64PhyAddr = pstImage->au64PhyAddr[2] + u64Offset;
    stSrc.u64VirAddr = pstImage->au64VirAddr[2] + u64Offset;
    stDst.u32Stride = pstDstImage->au32Stride[2];
    stDst.u64PhyAddr = pstDstImage->au64PhyAddr[2];
    stDst.u64VirAddr = pstDstImage->au64VirAddr[2];
    s32Ret = iveDmaCopy(&hIveHandle, &stSrc, &stDst, &stDmaCtrl, HI_TRUE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "iveDmaCopy 2 fail!\n");
    return s32Ret;

FAIL_0:
    freeIveImage(pstDstImage);
    return s32Ret;
}

static HI_VOID cropImageByIVE(const IVE_IMAGE_S* pstImage, const BondingBox_s *pstTightBbox,
    IVE_IMAGE_S* pstTarget, BondingBox_s* pstLocationBbox,
    HI_DOUBLE* pEdgeX, HI_DOUBLE* pEdgeY)
{
    HI_S32 s32Ret = HI_FAILURE;

    computeCropLocation(pstTightBbox, (HI_DOUBLE)pstImage->u32Width, (HI_DOUBLE)pstImage->u32Height, pstLocationBbox);

    const HI_DOUBLE roi_left = min(pstLocationBbox->x1, (HI_DOUBLE)(pstImage->u32Width - 1));
    const HI_DOUBLE roi_bottom = min(pstLocationBbox->y1, (HI_DOUBLE)(pstImage->u32Height - 1));
    const HI_DOUBLE roi_width = min((HI_DOUBLE)(pstImage->u32Width), max(1.0, ceil(pstLocationBbox->x2 - pstLocationBbox->x1)));
    const HI_DOUBLE roi_height = min((HI_DOUBLE)(pstImage->u32Height), max(1.0, ceil(pstLocationBbox->y2 - pstLocationBbox->y1)));

    IVE_IMAGE_S stCroppedImage = {0};
    IVE_RECT_U16_S stRect = {0};
    stRect.u16X = (HI_U16)roi_left;
    stRect.u16Y = (HI_U16)roi_bottom;
    stRect.u16Width = (HI_U16)roi_width;
    stRect.u16Height = (HI_U16)roi_height;
    s32Ret = createRectImageByIVE(pstImage, &stRect, &stCroppedImage);
    SAMPLE_CHK_RETURN_VOID(HI_SUCCESS != s32Ret, "rectImageByIVE fail!\n");

    const HI_DOUBLE output_width = max(ceil(compute_output_w(pstTightBbox->x1, pstTightBbox->x2)), roi_width);
    const HI_DOUBLE output_height = max(ceil(compute_output_h(pstTightBbox->y1, pstTightBbox->y2)), roi_height);

    *pEdgeX = min(compute_edge_x(pstTightBbox->x1, pstTightBbox->x2), (HI_DOUBLE)(output_width - 1));
    *pEdgeY = min(compute_edge_y(pstTightBbox->y1, pstTightBbox->y2), (HI_DOUBLE)(output_height - 1));

    IVE_IMAGE_S stOutputImage = {0};
    s32Ret = createImageByIVE(&stOutputImage, pstImage->enType, (HI_U32)output_width, (HI_U32)output_height);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_LOG_PRINT("createImageByIVE fail\n");
        freeIveImage(&stCroppedImage);
        return;
    }
    setIveImage(&stOutputImage, 0, 0, 0);

    IVE_RECT_U16_S stOutputRect;
    stOutputRect.u16X = (HI_U16)(*pEdgeX);
    stOutputRect.u16Y = (HI_U16)(*pEdgeY);
    stOutputRect.u16Width = (HI_U16)roi_width;
    stOutputRect.u16Height = (HI_U16)roi_height;
    s32Ret = copyImageByIVE(&stCroppedImage, &stOutputImage, &stOutputRect);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_LOG_PRINT("copyImageByIVE fail!\n");
        freeIveImage(&stCroppedImage);
        freeIveImage(&stOutputImage);
        return;
    }

    memcpy((HI_CHAR*)pstTarget, (HI_CHAR*)&stOutputImage, sizeof(IVE_IMAGE_S));
    freeIveImage(&stCroppedImage);
    return;
}

static HI_S32 subImageByIVE(IVE_IMAGE_S *pstSrcImage, HI_U8 B, HI_U8 G, HI_U8 R, IVE_IMAGE_S *pstDstImage)
{
    SAMPLE_CHK_RETURN((IVE_IMAGE_TYPE_U8C3_PLANAR != pstSrcImage->enType), HI_FAILURE, "Not support image type[%u]\n", pstSrcImage->enType);

    HI_S32 s32Ret = HI_FAILURE;
    HI_BOOL bFinish = HI_FALSE;
    IVE_HANDLE hIveHandle;
    IVE_IMAGE_S stImageC1 = {0};
    IVE_IMAGE_S stImageC2 = {0};
    IVE_IMAGE_S stImageC3 = {0};

    IVE_IMAGE_S stMeanC1 = {0};
    IVE_IMAGE_S stMeanC2 = {0};
    IVE_IMAGE_S stMeanC3 = {0};

    IVE_IMAGE_S stDstImageC1 = {0};
    IVE_IMAGE_S stDstImageC2 = {0};
    IVE_IMAGE_S stDstImageC3 = {0};

    IVE_SUB_CTRL_S stSubCtrl;
    stSubCtrl.enMode = IVE_SUB_MODE_ABS;
    struct timespec request_time;

    request_time.tv_sec = 0;
    request_time.tv_nsec = 100 * 1000 * 1000;

    HI_U32 u32OneCBytes = pstSrcImage->au32Stride[0] * pstSrcImage->u32Height;

    stImageC1.enType = IVE_IMAGE_TYPE_U8C1;
    stImageC2.enType = IVE_IMAGE_TYPE_U8C1;
    stImageC3.enType = IVE_IMAGE_TYPE_U8C1;
    stImageC1.u32Height = pstSrcImage->u32Height;
    stImageC2.u32Height = pstSrcImage->u32Height;
    stImageC3.u32Height = pstSrcImage->u32Height;
    stImageC1.u32Width = pstSrcImage->u32Width;
    stImageC2.u32Width = pstSrcImage->u32Width;
    stImageC3.u32Width = pstSrcImage->u32Width;
    stImageC1.au32Stride[0] = pstSrcImage->au32Stride[0];
    stImageC2.au32Stride[0] = pstSrcImage->au32Stride[1];
    stImageC3.au32Stride[0] = pstSrcImage->au32Stride[2];
    stImageC1.au64PhyAddr[0] = pstSrcImage->au64PhyAddr[0];
    stImageC2.au64PhyAddr[0] = pstSrcImage->au64PhyAddr[1];
    stImageC3.au64PhyAddr[0] = pstSrcImage->au64PhyAddr[2];
    stImageC1.au64VirAddr[0] = pstSrcImage->au64VirAddr[0];
    stImageC2.au64VirAddr[0] = pstSrcImage->au64VirAddr[1];
    stImageC3.au64VirAddr[0] = pstSrcImage->au64VirAddr[2];

    stDstImageC1.enType = IVE_IMAGE_TYPE_U8C1;
    stDstImageC2.enType = IVE_IMAGE_TYPE_U8C1;
    stDstImageC3.enType = IVE_IMAGE_TYPE_U8C1;
    stDstImageC1.u32Height = pstDstImage->u32Height;
    stDstImageC2.u32Height = pstDstImage->u32Height;
    stDstImageC3.u32Height = pstDstImage->u32Height;
    stDstImageC1.u32Width = pstDstImage->u32Width;
    stDstImageC2.u32Width = pstDstImage->u32Width;
    stDstImageC3.u32Width = pstDstImage->u32Width;
    stDstImageC1.au32Stride[0] = pstDstImage->au32Stride[0];
    stDstImageC2.au32Stride[0] = pstDstImage->au32Stride[1];
    stDstImageC3.au32Stride[0] = pstDstImage->au32Stride[2];
    stDstImageC1.au64PhyAddr[0] = pstDstImage->au64PhyAddr[0];
    stDstImageC2.au64PhyAddr[0] = pstDstImage->au64PhyAddr[1];
    stDstImageC3.au64PhyAddr[0] = pstDstImage->au64PhyAddr[2];
    stDstImageC1.au64VirAddr[0] = pstDstImage->au64VirAddr[0];
    stDstImageC2.au64VirAddr[0] = pstDstImage->au64VirAddr[1];
    stDstImageC3.au64VirAddr[0] = pstDstImage->au64VirAddr[2];


    s32Ret = createImageByIVE(&stMeanC1, IVE_IMAGE_TYPE_U8C1, pstSrcImage->u32Width, pstSrcImage->u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail:%#x\n", s32Ret);
    memset((HI_CHAR *)((HI_UL)(stMeanC1.au64VirAddr[0])), B, u32OneCBytes);

    s32Ret = createImageByIVE(&stMeanC2, IVE_IMAGE_TYPE_U8C1, pstSrcImage->u32Width, pstSrcImage->u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail:%#x\n", s32Ret);
    memset((HI_CHAR *)((HI_UL)(stMeanC2.au64VirAddr[0])), G, u32OneCBytes);

    s32Ret = createImageByIVE(&stMeanC3, IVE_IMAGE_TYPE_U8C1, pstSrcImage->u32Width, pstSrcImage->u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail:%#x\n", s32Ret);
    memset((HI_CHAR *)((HI_UL)(stMeanC3.au64VirAddr[0])), R, u32OneCBytes);

    s32Ret = HI_MPI_IVE_Sub(&hIveHandle, &stImageC1, &stMeanC1, &stDstImageC1, &stSubCtrl, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_MPI_IVE_Sub fail:%#x\n", s32Ret);

    s32Ret = HI_MPI_IVE_Sub(&hIveHandle, &stImageC2, &stMeanC2, &stDstImageC2, &stSubCtrl, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_MPI_IVE_Sub fail:%#x\n", s32Ret);

    s32Ret = HI_MPI_IVE_Sub(&hIveHandle, &stImageC3, &stMeanC3, &stDstImageC3, &stSubCtrl, HI_TRUE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_MPI_IVE_Sub fail:%#x\n", s32Ret);

    s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
    while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
    {
        nanosleep(&request_time, HI_NULL);
        printf("HI_MPI_IVE_Query timeout, wait 100us\n");
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
    }
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "HI_MPI_IVE_Query fail:%#x\n", s32Ret);

FAIL_0:
    freeIveImage(&stMeanC1);
    freeIveImage(&stMeanC2);
    freeIveImage(&stMeanC3);

    return s32Ret;
}

HI_S32 resizeImageByIVE(IVE_IMAGE_S *pstSrcImage, IVE_IMAGE_S *pstDstImage)
{
    HI_S32 s32Ret = HI_SUCCESS;
    IVE_RESIZE_CTRL_S stResizeCtrl = {0};
    HI_RUNTIME_MEM_S stResizeMem = {0};
    IVE_HANDLE hIveHandle;
    HI_BOOL bFinish = HI_FALSE;
    struct timespec request_time;

    request_time.tv_sec = 0;
    request_time.tv_nsec = 100 * 1000 * 1000;

    if ((0 != pstSrcImage->u32Height % 2) || (0 != pstSrcImage->u32Width % 2))
    {
        SAMPLE_LOG_PRINT("Invalid input image size for input, must be a multiply of 2\n");
        return HI_FAILURE;
    }

    stResizeCtrl.enMode = IVE_RESIZE_MODE_LINEAR;
    stResizeCtrl.u16Num = 1;
    stResizeCtrl.stMem.u32Size = 49 * stResizeCtrl.u16Num;
    stResizeMem.u32Size = stResizeCtrl.stMem.u32Size;
    s32Ret = SAMPLE_AllocMem(&stResizeMem, HI_FALSE);
    stResizeCtrl.stMem.u64PhyAddr = stResizeMem.u64PhyAddr;
    stResizeCtrl.stMem.u64VirAddr = stResizeMem.u64VirAddr;
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "SAMPLE_Utils_AllocMem fail:%#x", s32Ret);

    s32Ret = HI_MPI_IVE_Resize(&hIveHandle, pstSrcImage, pstDstImage, &stResizeCtrl, HI_TRUE);
    SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "HI_MPI_IVE_Resize fail:%#x", s32Ret);

    s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
    while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
    {
        nanosleep(&request_time, HI_NULL);
        printf("HI_MPI_IVE_Query timeout, wait 100us\n");
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, HI_TRUE);
    }
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "Error(%#x), HI_MPI_IVE_Query failed!\n", s32Ret);
FAIL_0:
    SAMPLE_FreeMem(&stResizeMem);

    return s32Ret;
}

static HI_S32 meanImageByIVE(IVE_IMAGE_S *pstSrcImage, IVE_IMAGE_S* pstDstImage)
{
    HI_S32 s32Ret = HI_FAILURE;
    IVE_IMAGE_S stTempMean = {0};
    IVE_RECT_U16_S stDstRect = {0};
    HI_U32 u32Width = 0;
    HI_U32 u32Height = 0;
    IVE_IMAGE_S stTempDst = {0};

    if ((0 != pstSrcImage->u32Height % 2) || (0 != pstSrcImage->u32Width % 2))
    {
        // 227*227--->228*228---->227*227
        u32Width = pstSrcImage->u32Width;
        u32Height = pstSrcImage->u32Height;

        s32Ret = createImageByIVE(&stTempMean, pstSrcImage->enType, u32Width + u32Width % 2, u32Height + u32Height % 2);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "createImageByIVE 1 fail:%#x", s32Ret);

        stDstRect.u16X = 0;
        stDstRect.u16Y = 0;
        stDstRect.u16Width = u32Width;
        stDstRect.u16Height = u32Height;
        s32Ret = copyImageByIVE(pstSrcImage, &stTempMean, &stDstRect);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "copyImageByIVE 1 fail:%#x", s32Ret);

        if ((stTempMean.u32Height != pstDstImage->u32Height) || (stTempMean.u32Width != pstDstImage->u32Width))
        {
            s32Ret = createImageByIVE(&stTempDst, pstSrcImage->enType, u32Width + u32Width % 2, u32Height + u32Height % 2);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "createImageByIVE 2 fail:%#x", s32Ret);

            s32Ret = subImageByIVE(&stTempMean, 104, 117, 123, &stTempDst);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "subImageByIVE 1 fail:%#x", s32Ret);

            stDstRect.u16X = 0;
            stDstRect.u16Y = 0;
            stDstRect.u16Width = pstDstImage->u32Width;
            stDstRect.u16Height = pstDstImage->u32Height;
            s32Ret = copyImageByIVE(&stTempMean, pstDstImage, &stDstRect);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "copyImageByIVE 2 fail:%#x", s32Ret);

        }
        else
        {
            s32Ret = subImageByIVE(&stTempMean, 104, 117, 123, pstDstImage);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "subImageByIVE 2 fail:%#x", s32Ret);
        }
    }
    else
    {
        // 228*228--->228*228---->227*227
        if ((pstDstImage->u32Height != pstSrcImage->u32Height) || (pstDstImage->u32Width != pstSrcImage->u32Width))
        {
            s32Ret = createImageByIVE(&stTempDst, pstSrcImage->enType, pstSrcImage->u32Width, pstSrcImage->u32Height);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "createImageByIVE 3 fail:%#x", s32Ret);

            s32Ret = subImageByIVE(pstSrcImage, 104, 117, 123, &stTempDst);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "subImageByIVE 3 fail:%#x", s32Ret);

            stDstRect.u16X = 0;
            stDstRect.u16Y = 0;
            stDstRect.u16Width = pstDstImage->u32Width;
            stDstRect.u16Height = pstDstImage->u32Height;
            s32Ret = copyImageByIVE(&stTempDst, pstDstImage, &stDstRect);
            SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "copyImageByIVE 3 fail:%#x", s32Ret);
        }
        else
        {
            s32Ret = subImageByIVE(pstSrcImage, 104, 117, 123, pstDstImage);
            SAMPLE_CHK_GOTO((HI_SUCCESS), FAIL_0, "subImageByIVE 4 fail:%#x", s32Ret);
        }
    }

FAIL_0:
    freeIveImage(&stTempMean);
    freeIveImage(&stTempDst);
    return s32Ret;
}

static HI_S32 goTurnImagePreprocessByIVE(IVE_IMAGE_S* pstSrcImage, IVE_IMAGE_S* pstDstImage)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32Height = pstDstImage->u32Height;
    HI_U32 u32Width = pstDstImage->u32Width;
    IVE_IMAGE_S stResizeImage = {0};

    if ((pstSrcImage->u32Height != u32Height) || (pstSrcImage->u32Width != u32Width))
    {
        s32Ret = createImageByIVE(&stResizeImage, pstSrcImage->enType, u32Width + u32Width % 2, u32Height + u32Height % 2);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "createImageByIVE fail:%#x\n", s32Ret);

        s32Ret = resizeImageByIVE(pstSrcImage, &stResizeImage);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "resizeImageByIVE fail:%#x\n", s32Ret);

        s32Ret = meanImageByIVE(&stResizeImage, pstDstImage);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "meanImageByIVE fail:%#x\n", s32Ret);

    }
    else
    {
        s32Ret = meanImageByIVE(pstSrcImage, pstDstImage);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "meanImageByIVE fail:%#x\n", s32Ret);
    }

FAIL_0:
    freeIveImage(&stResizeImage);
    return s32Ret;
}

static HI_S32 goTurnReadInputDataByIVE(IVE_IMAGE_S* pstSrcImage, HI_RUNTIME_BLOB_S *pstBlobs, HI_U16 index)
{
    HI_S32 s32Ret = HI_FAILURE;
    IVE_IMAGE_S stDstImage = {0};

    s32Ret = createImageByIVE(&stDstImage, pstSrcImage->enType, pstBlobs->unShape.stWhc.u32Width, pstBlobs->unShape.stWhc.u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail:%#x\n", s32Ret);
    setIveImage(&stDstImage, 100, 100, 100);

    s32Ret = goTurnImagePreprocessByIVE(pstSrcImage, &stDstImage);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "goTurnImagePreprocessByIVE fail:%#x\n", s32Ret);

    s32Ret = copyIveImage2Blob(&stDstImage, pstBlobs, index);
FAIL_0:
    freeIveImage(&stDstImage);
    return s32Ret;
}

HI_S32 cropPadBlobByIVE(const HI_RUNTIME_BLOB_S* pstSrcBlob,
    const HI_RUNTIME_BLOB_S* pstBboxBlob,
    HI_RUNTIME_BLOB_S* pstDstBlob,
    TRACKER_LOCATION_S* pstLocation,
    HI_S32* ps32EdgeX,
    HI_S32* ps32EdgeY,
    HI_S32* ps32RegionW,
    HI_S32* ps32RegionH)
{
    IVE_IMAGE_S stImage = {0};
    IVE_IMAGE_S stTarget = {0};
    HI_S32 s32Ret = HI_FAILURE;

    if (4 != pstBboxBlob->unShape.stWhc.u32Width)
    {
        printf("Error bbox blob u32Num[%u]", pstBboxBlob->u32Num);
        return HI_FAILURE;
    }

    s32Ret = createImageByIVE(&stImage, IVE_IMAGE_TYPE_U8C3_PLANAR, pstSrcBlob->unShape.stWhc.u32Width, pstSrcBlob->unShape.stWhc.u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail\n");

    s32Ret = copyBlob2IveImage(pstSrcBlob, 0, &stImage);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "copyBlob2IveImage fail\n");

    BondingBox_s stTightBbox;
    BondingBox_s stBboxLocation;
    HI_DOUBLE dEdgeX = 0;
    HI_DOUBLE dEdgeY = 0;

    HI_U32 u32OffSet = pstBboxBlob->unShape.stWhc.u32Chn * pstBboxBlob->unShape.stWhc.u32Height * pstBboxBlob->u32Stride;
    for (HI_U32 i = 0; i < pstBboxBlob->u32Num; i++)
    {
        HI_S32 *ps32Dst = (HI_S32 *)((HI_U8 *)((HI_UL)(pstBboxBlob->u64VirAddr)) + i * u32OffSet);
        stTightBbox.x1 = (HI_DOUBLE)*ps32Dst;
        stTightBbox.y1 = (HI_DOUBLE)*(ps32Dst + 1);
        stTightBbox.x2 = (HI_DOUBLE)*(ps32Dst + 2);
        stTightBbox.y2 = (HI_DOUBLE)*(ps32Dst + 3);

        cropImageByIVE(&stImage, &stTightBbox, &stTarget, &stBboxLocation, &dEdgeX, &dEdgeY);
        SAMPLE_CHK_GOTO(0 == stTarget.u32Height, FAIL_0, "cropImageByIVE fail!\n");

        goTurnReadInputDataByIVE(&stTarget, pstDstBlob, i);

        pstLocation[i].x1 = (HI_S32)(stBboxLocation.x1 * 4096);
        pstLocation[i].y1 = (HI_S32)(stBboxLocation.y1 * 4096);
        pstLocation[i].x2 = (HI_S32)(stBboxLocation.x2 * 4096);
        pstLocation[i].y2 = (HI_S32)(stBboxLocation.y2 * 4096);
        ps32EdgeX[i] = (HI_S32)(dEdgeX * 4096);
        ps32EdgeY[i] = (HI_S32)(dEdgeY * 4096);
        ps32RegionW[i] = stTarget.u32Width;
        ps32RegionH[i] = stTarget.u32Height;

        printf("location %d: (%f,%f)(%f,%f)\n", i, stBboxLocation.x1, stBboxLocation.y1, stBboxLocation.x2, stBboxLocation.y2);
        printf("edge %d: %f,%f ; %d,%d \n", i, dEdgeX, dEdgeY, stTarget.u32Width, stTarget.u32Height);
        freeIveImage(&stTarget);

    }

FAIL_0:
    freeIveImage(&stImage);

    return HI_SUCCESS;
}

HI_S32 resizeByIVE(const HI_RUNTIME_BLOB_S *pstSrcBlob, HI_RUNTIME_BLOB_S* pstDstBlob)
{
    HI_S32 s32Ret = HI_FAILURE;
    IVE_IMAGE_S stSrcImage = {0};
    IVE_IMAGE_S stDstImage = {0};
    IVE_IMAGE_S stDstTemp = {0};
    IVE_RECT_U16_S stDstRect = {0};
    HI_U32 u32Width = pstDstBlob->unShape.stWhc.u32Width + pstDstBlob->unShape.stWhc.u32Width % 2;
    HI_U32 u32Height = pstDstBlob->unShape.stWhc.u32Height + pstDstBlob->unShape.stWhc.u32Height % 2;

    s32Ret = createImageByIVE(&stSrcImage, IVE_IMAGE_TYPE_U8C3_PLANAR, pstSrcBlob->unShape.stWhc.u32Width, pstSrcBlob->unShape.stWhc.u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail!\n");

    s32Ret = createImageByIVE(&stDstImage, IVE_IMAGE_TYPE_U8C3_PLANAR, u32Width, u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail!\n");

    s32Ret = copyBlob2IveImage(pstSrcBlob, 0, &stSrcImage);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "copyBlob2IveImage fail!\n");

    s32Ret = resizeImageByIVE(&stSrcImage, &stDstImage);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "resizeImageByIVE fail!\n");

    if ((stDstImage.u32Height != pstDstBlob->unShape.stWhc.u32Height)
        || (stDstImage.u32Width != pstDstBlob->unShape.stWhc.u32Width))
    {
        s32Ret = createImageByIVE(&stDstTemp, IVE_IMAGE_TYPE_U8C3_PLANAR, pstDstBlob->unShape.stWhc.u32Width, pstDstBlob->unShape.stWhc.u32Height);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail!\n");

        stDstRect.u16X = 0;
        stDstRect.u16Y = 0;
        stDstRect.u16Width = pstDstBlob->unShape.stWhc.u32Width;
        stDstRect.u16Height = pstDstBlob->unShape.stWhc.u32Height;
        s32Ret = copyImageByIVE(&stDstImage, &stDstTemp, &stDstRect);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "copyImageByIVE fail:%#x", s32Ret);

        s32Ret = copyIveImage2Blob(&stDstTemp, pstDstBlob, 0);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "copyIveImage2Blob fail:%#x", s32Ret);
    }
    else
    {
        s32Ret = copyIveImage2Blob(&stDstImage, pstDstBlob, 0);
        SAMPLE_CHK_GOTO((HI_SUCCESS != s32Ret), FAIL_0, "copyIveImage2Blob fail:%#x", s32Ret);
    }
FAIL_0:
    freeIveImage(&stSrcImage);
    freeIveImage(&stDstImage);
    freeIveImage(&stDstTemp);
    return s32Ret;
}


HI_VOID savePPMImage(const HI_CHAR* pszPicPath, IVE_IMAGE_S* pstImage)
{
    HI_S32 s32Ret = 0;
    HI_CHAR aszAbsDirPath[PATH_MAX + 1];
    HI_CHAR aszAbsFilePath[PATH_MAX + 1];
    HI_CHAR aszPicPathForDirName[PATH_MAX + 1];
    HI_CHAR aszPicPathForBaseName[PATH_MAX + 1];
    HI_CHAR* pszDirName = HI_NULL;
    HI_CHAR* pszBaseName = HI_NULL;

    SAMPLE_CHK_RETURN_VOID((HI_NULL == pszPicPath || strlen(pszPicPath) > PATH_MAX), "invalid pic path(%s) is NULL or too long", pszPicPath);

    memset(aszPicPathForDirName, 0, sizeof(aszPicPathForDirName));
    memset(aszPicPathForBaseName, 0, sizeof(aszPicPathForBaseName));

    strncpy(aszPicPathForDirName, pszPicPath, PATH_MAX);
    strncpy(aszPicPathForBaseName, pszPicPath, PATH_MAX);

    pszDirName = dirname(aszPicPathForDirName);
    SAMPLE_CHK_RETURN_VOID((HI_NULL == pszDirName || strlen(pszDirName) <= 0), "dirname(%s) error", pszPicPath);
    pszBaseName = basename(aszPicPathForBaseName);
    SAMPLE_CHK_RETURN_VOID((HI_NULL == pszBaseName || strlen(pszBaseName) <= 0), "basename(%s) error", pszPicPath);

    SAMPLE_CHK_RETURN_VOID(('.' == pszBaseName[strlen(pszBaseName) - 1] || '/' == pszBaseName[strlen(pszBaseName) - 1]),
            "invalid path[%s]: miss file name", pszPicPath);

    memset(aszAbsDirPath, 0, sizeof(aszAbsDirPath));
    SAMPLE_CHK_RETURN_VOID(HI_NULL == realpath(pszDirName, aszAbsDirPath), "realpath fail");

    SAMPLE_CHK_RETURN_VOID((strlen(aszAbsDirPath) + strlen(pszBaseName) + 1 > PATH_MAX), "aszAbsDirPath[%s%s] too long", aszAbsDirPath, pszBaseName);

    s32Ret = snprintf(aszAbsFilePath, sizeof(aszAbsFilePath), "%s/%s", aszAbsDirPath, pszBaseName);
    SAMPLE_CHK_RETURN_VOID(s32Ret <= 0, "snprintf absFileName fail");

    FILE *ppmFile = fopen(aszAbsFilePath, "wb");
    SAMPLE_CHK_RETURN_VOID(HI_NULL == ppmFile, "open file %s fail!\n", aszAbsFilePath);
    fprintf(ppmFile, "P6\n%d %d\n255\n", pstImage->u32Width, pstImage->u32Height);

    unsigned char * pucImageC0 = (unsigned char *)((HI_UL)pstImage->au64VirAddr[0]);
    unsigned char * pucImageC1 = (unsigned char *)((HI_UL)pstImage->au64VirAddr[1]);
    unsigned char * pucImageC2 = (unsigned char *)((HI_UL)pstImage->au64VirAddr[2]);

    unsigned char  aucPixel[3];
    for (HI_U32 i = 0; i < pstImage->u32Height; i++)
    {
        for (HI_U32 j = 0; j < pstImage->u32Width; j++)
        {
            aucPixel[2] = pucImageC0[i * pstImage->au32Stride[0] + j];
            aucPixel[1] = pucImageC1[i * pstImage->au32Stride[1] + j];
            aucPixel[0] = pucImageC2[i * pstImage->au32Stride[2] + j];

            s32Ret = fwrite(aucPixel, 3, 1, ppmFile);
            if (s32Ret != 1)
            {
                fclose(ppmFile);
                return;
            }
        }
    }
    fclose(ppmFile);
    return;
}

HI_VOID drawImageRectByIVE(const HI_CHAR* pszPicPath, const HI_RUNTIME_BLOB_S* pstBlob,
                         HI_S32 as32Coord[], HI_U32 u32CoordCnt, HI_U32 u32CoordStride)
{
    IVE_IMAGE_S stSrcImage = {0};
    IVE_IMAGE_S stRectImage = {0};
    IVE_IMAGE_S stBluePatch = {0};
    IVE_RECT_U16_S stRect = {0};
    HI_S32 s32Ret = HI_SUCCESS;

    s32Ret = createImageByIVE(&stSrcImage, IVE_IMAGE_TYPE_U8C3_PLANAR, pstBlob->unShape.stWhc.u32Width, pstBlob->unShape.stWhc.u32Height);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail!\n");

    s32Ret = copyBlob2IveImage(pstBlob, 0, &stSrcImage);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "copyBlob2IveImage fail!\n");

    for (HI_U32 i = 0; i < u32CoordCnt; i++)
    {
        stRect.u16X = as32Coord[i * u32CoordStride] + 1;
        stRect.u16Y = as32Coord[i * u32CoordStride + 1] + 1;
        stRect.u16Width = as32Coord[i * u32CoordStride + 2] - as32Coord[i * u32CoordStride] - 2;
        stRect.u16Height = as32Coord[i * u32CoordStride + 3] - as32Coord[i * u32CoordStride + 1] - 2;
        s32Ret = createRectImageByIVE(&stSrcImage, &stRect, &stRectImage);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createRectImageByIVE fail!\n");


        s32Ret = createImageByIVE(&stBluePatch, IVE_IMAGE_TYPE_U8C3_PLANAR,
            as32Coord[i * u32CoordStride + 2] - as32Coord[i * u32CoordStride], as32Coord[i * u32CoordStride + 3] - as32Coord[i * u32CoordStride + 1]);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "createImageByIVE fail!\n");

        setIveImage(&stBluePatch, 0, 0, 255);

        stRect.u16X = as32Coord[i * u32CoordStride];
        stRect.u16Y = as32Coord[i * u32CoordStride + 1];
        stRect.u16Width = as32Coord[i * u32CoordStride + 2] - as32Coord[i * u32CoordStride + 0];
        stRect.u16Height = as32Coord[i * u32CoordStride + 3] - as32Coord[i * u32CoordStride + 1];
        s32Ret = copyImageByIVE(&stBluePatch, &stSrcImage, &stRect);
        freeIveImage(&stBluePatch);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "copyImageByIVE fail!\n");

        stRect.u16X = as32Coord[i * u32CoordStride] + 1;
        stRect.u16Y = as32Coord[i * u32CoordStride + 1] + 1;
        stRect.u16Width = as32Coord[i * u32CoordStride + 2] - as32Coord[i * u32CoordStride] - 2;
        stRect.u16Height = as32Coord[i * u32CoordStride + 3] - as32Coord[i * u32CoordStride + 1] - 2;
        s32Ret = copyImageByIVE(&stRectImage, &stSrcImage, &stRect);
        freeIveImage(&stRectImage);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "copyImageByIVE fail!\n");

    }
    printf("save %s\n", pszPicPath);
    savePPMImage(pszPicPath, &stSrcImage);
FAIL_0:
    freeIveImage(&stSrcImage);
    freeIveImage(&stBluePatch);
    freeIveImage(&stRectImage);
    return;
}

HI_VOID saveBlobByIVE(const HI_CHAR* pszPath, const HI_RUNTIME_BLOB_S* pstSrcBlob, HI_U16 u16Index)
{
    IVE_IMAGE_S stImage = {0};
    HI_S32 s32Ret = HI_SUCCESS;
    s32Ret = copyBlob2IveImage(pstSrcBlob, u16Index, &stImage);
    if (HI_SUCCESS == s32Ret)
    {
        savePPMImage(pszPath, &stImage);
    }
    freeIveImage(&stImage);
}
#endif
