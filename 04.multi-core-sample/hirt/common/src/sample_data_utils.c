#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include "math.h"
#include "sample_log.h"
#include "sample_memory_ops.h"
#include "sample_save_blob.h"
#include "sample_resize_roi.h"
#include "sample_data_utils.h"

#ifdef _WIN32
#include <Windows.h>
#define BILLION                             (1E9)

static BOOL g_first_time = 1;
static LARGE_INTEGER g_counts_per_sec;
static int CLOCK_MONOTONIC = 1;

int clock_gettime(int type, struct timespec* ct)
{
    LARGE_INTEGER count;

    if (g_first_time)
    {
        g_first_time = 0;

        if (0 == QueryPerformanceFrequency(&g_counts_per_sec))
        {
            g_counts_per_sec.QuadPart = 0;
        }
    }

    if ((NULL == ct) || (g_counts_per_sec.QuadPart <= 0) ||
        (0 == QueryPerformanceCounter(&count)))
    {
        return -1;
    }

    ct->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
    ct->tv_nsec = (long)(((count.QuadPart % g_counts_per_sec.QuadPart) * BILLION) / g_counts_per_sec.QuadPart);

    return 0;
}
#endif

HI_VOID timeSpendMs(struct timespec* ptime1, struct timespec* ptime2, char* des)
{
    long spend;
    spend = ((ptime2->tv_sec - ptime1->tv_sec) * 1000 + (ptime2->tv_nsec - ptime1->tv_nsec) / 1000000);
    printf("[%s]===== TIME SPEND: %ld ms =====\n", des, spend);
}

HI_VOID timePrint(struct timespec* ptime, char* des)
{
    printf("\n[%s]===== TIME NOW: %ld s, %ld us ===== ", des, (ptime->tv_sec), (ptime->tv_nsec / 1000));
}

#define SAMPLE_RUNTIME_CHECK(cond, ec) \
    do{\
        if (!(cond)) {\
            printf("%s %d CHECK error! cond = %d, do ret = %d\n", __FILE__, __LINE__, cond, ec);\
            return ec;\
        }\
    }while(0)

typedef struct hiNNIE_STACK
{
    HI_S32 s32Min;
    HI_S32 s32Max;
} NNIE_STACK_S;

/*SSD software parameter*/
typedef struct hiSAMPLE_SSD_PARAM_S
{
    /*----------------- Model Parameters ---------------*/
    HI_U32 au32ConvHeight[12];
    HI_U32 au32ConvWidth[12];
    HI_U32 au32ConvChannel[12];
    /*----------------- PriorBox Parameters ---------------*/
    HI_U32 au32PriorBoxWidth[6];
    HI_U32 au32PriorBoxHeight[6];
    HI_FLOAT af32PriorBoxMinSize[6][1];
    HI_FLOAT af32PriorBoxMaxSize[6][1];
    HI_U32 u32MinSizeNum;
    HI_U32 u32MaxSizeNum;
    HI_U32 u32OriImHeight;
    HI_U32 u32OriImWidth;
    HI_U32 au32InputAspectRatioNum[6];
    HI_FLOAT af32PriorBoxAspectRatio[6][2];
    HI_FLOAT af32PriorBoxStepWidth[6];
    HI_FLOAT af32PriorBoxStepHeight[6];
    HI_FLOAT f32Offset;
    HI_BOOL bFlip;
    HI_BOOL bClip;
    HI_S32 as32PriorBoxVar[4];
    /*----------------- Softmax Parameters ---------------*/
    HI_U32 au32SoftMaxInChn[6];
    HI_U32 u32SoftMaxInHeight;
    HI_U32 u32ConcatNum;
    HI_U32 u32SoftMaxOutWidth;
    HI_U32 u32SoftMaxOutHeight;
    HI_U32 u32SoftMaxOutChn;
    /*----------------- DetectionOut Parameters ---------------*/
    HI_U32 u32ClassNum;
    HI_U32 u32TopK;
    HI_U32 u32KeepTopK;
    HI_U32 u32NmsThresh;
    HI_U32 u32ConfThresh;
    HI_U32 au32DetectInputChn[6];
    HI_U32 au32ConvStride[6];
    HI_RUNTIME_MEM_S stPriorBoxTmpBuf;
    HI_RUNTIME_MEM_S stSoftMaxTmpBuf;
    HI_RUNTIME_MEM_S stGetResultTmpBuf;
    HI_RUNTIME_MEM_S stClassRoiNum;
    HI_RUNTIME_MEM_S stDstRoi;
    HI_RUNTIME_MEM_S stDstScore;
} HI_SAMPLE_SSD_PARAM_S;

HI_VOID SAMPLE_DATA_GetStride(HI_RUNTIME_BLOB_TYPE_E type, HI_U32 width, HI_U32 align, HI_U32* pStride)
{
    HI_U32 u32Size = 0;

    if (HI_RUNTIME_BLOB_TYPE_S32 == type || HI_RUNTIME_BLOB_TYPE_VEC_S32 == type || HI_RUNTIME_BLOB_TYPE_SEQ_S32 == type)
    {
        u32Size = sizeof(HI_U32);
    }
    else
    {
        u32Size = sizeof(HI_U8);
    }

#if 0

    if (HI_RUNTIME_BLOB_TYPE_SEQ_S32 == type)
    {
        if (ALIGN_16 == align)
        {
            *pStride = ALIGN16(u32Dim * u32Size);
        }
        else
        {
            *pStride = ALIGN32(u32Dim * u32Size);
        }

        total = step * *pStride;
    }
    else
#endif
    {
        if (ALIGN_16 == align)
        {
            *pStride = ALIGN16(width * u32Size);
        }
        else
        {
            *pStride = ALIGN32(width * u32Size);
        }
    }

    return;
}

HI_U32 SAMPLE_DATA_GetBlobSize(HI_U32 stride, HI_U32 num, HI_U32 height, HI_U32 chn)
{
    return num * stride * height * chn;
}

HI_VOID printDebugData(const HI_CHAR* pcName, HI_U64 u64VirAddr, HI_U32 u32PrintLine)
{
    HI_U8* pu8Tmp = HI_NULL;
    printf("============================== %s result print =============================\n", pcName);
    pu8Tmp = (HI_U8*)((HI_UL)(u64VirAddr));

    for (HI_U32 i = 0; i < u32PrintLine; i++)
    {
        for (HI_U32 j = 0; j < 16; j++)
        {
            printf("%02x ", pu8Tmp[i * 16 + j]);
        }

        printf("\n");
    }

    printf("============================== %s result end =============================\n", pcName);
}

static HI_S32 SizeClip(HI_S32 s32inputSize, HI_S32 s32sizeMin, HI_S32 s32sizeMax)
{
    return max(min(s32inputSize, s32sizeMax), s32sizeMin);
}

static HI_S32 BboxClip(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH)
{
    ps32Proposals[0] = SizeClip(ps32Proposals[0], 0, (HI_S32)u32ImageW - 1);
    ps32Proposals[1] = SizeClip(ps32Proposals[1], 0, (HI_S32)u32ImageH - 1);
    ps32Proposals[2] = SizeClip(ps32Proposals[2], 0, (HI_S32)u32ImageW - 1);
    ps32Proposals[3] = SizeClip(ps32Proposals[3], 0, (HI_S32)u32ImageH - 1);

    return HI_SUCCESS;
}

static HI_S32 BboxClip_N(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH, HI_U32 u32Num)
{
    HI_S32 s32Ret = HI_FAILURE;

    for (HI_U32 i = 0; i < u32Num; i++)
    {
        s32Ret = BboxClip(&ps32Proposals[i * SVP_WK_PROPOSAL_WIDTH], u32ImageW, u32ImageH);
        SAMPLE_RUNTIME_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);
    }

    return HI_SUCCESS;
}

static HI_S32 Argswap(HI_S32* ps32Src1, HI_S32* ps32Src2)
{
    HI_U32 i = 0;
    HI_S32 tmp = 0;

    for (i = 0; i < SVP_WK_PROPOSAL_WIDTH; i++)
    {
        tmp = ps32Src1[i];
        ps32Src1[i] = ps32Src2[i];
        ps32Src2[i] = tmp;
    }

    return HI_SUCCESS;
}

static HI_S32 NonRecursiveArgQuickSort(HI_S32* aResultArray,
                                       HI_S32 s32Low, HI_S32 s32High, NNIE_STACK_S* pstStack, HI_U32 u32MaxNum)
{
    HI_S32 i = s32Low;
    HI_S32 j = s32High;
    HI_S32 s32Top = 0;
    pstStack[s32Top].s32Min = s32Low;
    pstStack[s32Top].s32Max = s32High;

    HI_S32 s32KeyConfidence = aResultArray[SVP_WK_PROPOSAL_WIDTH * s32Low + 4];

    while (s32Top > -1)
    {
        s32Low = pstStack[s32Top].s32Min;
        s32High = pstStack[s32Top].s32Max;
        i = s32Low;
        j = s32High;
        s32Top--;

        s32KeyConfidence = aResultArray[SVP_WK_PROPOSAL_WIDTH * s32Low + 4];

        while (i < j)
        {
            while ((i < j) && (s32KeyConfidence > aResultArray[j * SVP_WK_PROPOSAL_WIDTH + 4]))
            {
                j--;
            }

            if (i < j)
            {
                Argswap(&aResultArray[i * SVP_WK_PROPOSAL_WIDTH], &aResultArray[j * SVP_WK_PROPOSAL_WIDTH]);
                i++;
            }

            while ((i < j) && (s32KeyConfidence < aResultArray[i * SVP_WK_PROPOSAL_WIDTH + 4]))
            {
                i++;
            }

            if (i < j)
            {
                Argswap(&aResultArray[i * SVP_WK_PROPOSAL_WIDTH], &aResultArray[j * SVP_WK_PROPOSAL_WIDTH]);
                j--;
            }
        }

        if (s32Low <= u32MaxNum)
        {
            if (s32Low < i - 1)
            {
                s32Top++;
                pstStack[s32Top].s32Min = s32Low;
                pstStack[s32Top].s32Max = i - 1;
            }

            if (s32High > i + 1)
            {
                s32Top++;
                pstStack[s32Top].s32Min = i + 1;
                pstStack[s32Top].s32Max = s32High;
            }

            if (s32Top > MAX_STACK_DEPTH)
            {
                return HI_FAILURE;
            }
        }
    }

    return HI_SUCCESS;
}

static HI_S32 Overlap(HI_S32 s32XMin1, HI_S32 s32YMin1, HI_S32 s32XMax1, HI_S32 s32YMax1, HI_S32 s32XMin2,
                      HI_S32 s32YMin2, HI_S32 s32XMax2, HI_S32 s32YMax2, HI_S32* s32AreaSum, HI_S32* s32AreaInter)
{
    /*** Check the input, and change the Return value  ***/
    HI_S32 s32Inter = 0;
    HI_S32 s32Total = 0;
    HI_S32 s32XMin = 0;
    HI_S32 s32YMin = 0;
    HI_S32 s32XMax = 0;
    HI_S32 s32YMax = 0;
    HI_S32 s32Area1 = 0;
    HI_S32 s32Area2 = 0;
    HI_S32 s32InterWidth = 0;
    HI_S32 s32InterHeight = 0;

    s32XMin = max(s32XMin1, s32XMin2);
    s32YMin = max(s32YMin1, s32YMin2);
    s32XMax = min(s32XMax1, s32XMax2);
    s32YMax = min(s32YMax1, s32YMax2);

    s32InterWidth = s32XMax - s32XMin + 1;
    s32InterHeight = s32YMax - s32YMin + 1;

    s32InterWidth = (s32InterWidth >= 0) ? s32InterWidth : 0;
    s32InterHeight = (s32InterHeight >= 0) ? s32InterHeight : 0;

    s32Inter = s32InterWidth * s32InterHeight;
    s32Area1 = (s32XMax1 - s32XMin1 + 1) * (s32YMax1 - s32YMin1 + 1);
    s32Area2 = (s32XMax2 - s32XMin2 + 1) * (s32YMax2 - s32YMin2 + 1);

    s32Total = s32Area1 + s32Area2 - s32Inter;

    *s32AreaSum = s32Total;
    *s32AreaInter = s32Inter;

    return HI_SUCCESS;
}

static HI_S32 NonMaxSuppression(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh)
{
    /****** define variables *******/
    HI_S32 s32XMin1 = 0;
    HI_S32 s32YMin1 = 0;
    HI_S32 s32XMax1 = 0;
    HI_S32 s32YMax1 = 0;
    HI_S32 s32XMin2 = 0;
    HI_S32 s32YMin2 = 0;
    HI_S32 s32XMax2 = 0;
    HI_S32 s32YMax2 = 0;
    HI_S32 s32AreaTotal = 0;
    HI_S32 s32AreaInter = 0;

    for (HI_U32 i = 0; i < u32NumAnchors; i++)
    {
        if (RPN_SUPPRESS_FALSE == pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5])
        {
            s32XMin1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i];
            s32YMin1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 1];
            s32XMax1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 2];
            s32YMax1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 3];

            for (HI_U32 j = i + 1; j < u32NumAnchors; j++)
            {
                if (RPN_SUPPRESS_FALSE == pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j + 5])
                {
                    s32XMin2 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j];
                    s32YMin2 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j + 1];
                    s32XMax2 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j + 2];
                    s32YMax2 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j + 3];

                    if((s32XMin2>s32XMax1) || (s32XMax2<s32XMin1) || (s32YMin2>s32YMax1) || (s32YMax2<s32YMin1))
                    {
                        continue;
                    }

                    Overlap(s32XMin1, s32YMin1, s32XMax1, s32YMax1, s32XMin2, s32YMin2, s32XMax2, s32YMax2, &s32AreaTotal, &s32AreaInter);

                    if (s32AreaInter * SVP_WK_QUANT_BASE > (HI_S32)u32NmsThresh * s32AreaTotal)
                    {
                        if (pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 4] >= pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j + 4])
                        {
                            pu32Proposals[SVP_WK_PROPOSAL_WIDTH * j + 5] = RPN_SUPPRESS_TRUE;
                        }

                        else
                        {
                            pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5] = RPN_SUPPRESS_TRUE;
                        }
                    }
                }
            }
        }
    }

    return HI_SUCCESS;
}

static HI_U32 SAMPLE_Ssd_GetResultTmpBuf(HI_SAMPLE_SSD_PARAM_S* pstSsdParam)
{
    HI_U32 u32PriorBoxSize = 0;
    HI_U32 u32SoftMaxSize = 0;
    HI_U32 u32DetectionSize = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32PriorNum = 0;
    HI_U32 i = 0;

    /*priorbox size*/
    for(i = 0; i < 6; i++)
    {
        u32PriorBoxSize += pstSsdParam->au32PriorBoxHeight[i]*pstSsdParam->au32PriorBoxWidth[i]*
            SVP_WK_COORDI_NUM*2*(pstSsdParam->u32MaxSizeNum+pstSsdParam->u32MinSizeNum+
            pstSsdParam->au32InputAspectRatioNum[i]*2*pstSsdParam->u32MinSizeNum)*sizeof(HI_U32);
    }
    pstSsdParam->stPriorBoxTmpBuf.u32Size = u32PriorBoxSize;
    u32TotalSize+=u32PriorBoxSize;
    /*softmax size*/
    for(i = 0; i < pstSsdParam->u32ConcatNum; i++)
    {
        u32SoftMaxSize += pstSsdParam->au32SoftMaxInChn[i]*sizeof(HI_U32);
    }
    pstSsdParam->stSoftMaxTmpBuf.u32Size = u32SoftMaxSize;
    u32TotalSize+=u32SoftMaxSize;

    /*detection size*/
    for(i = 0; i < pstSsdParam->u32ConcatNum; i++)
    {
        u32PriorNum+=pstSsdParam->au32DetectInputChn[i]/SVP_WK_COORDI_NUM;
    }
    u32DetectionSize+=u32PriorNum*SVP_WK_COORDI_NUM*sizeof(HI_U32);
    u32DetectionSize+=u32PriorNum*SVP_WK_PROPOSAL_WIDTH*sizeof(HI_U32)*2;
    u32DetectionSize+=u32PriorNum*2*sizeof(HI_U32);
    pstSsdParam->stGetResultTmpBuf.u32Size = u32DetectionSize;
    u32TotalSize+=u32DetectionSize;

    return u32TotalSize;
}

static HI_S32 SAMPLE_Ssd_InitParam(HI_RUNTIME_BLOB_S* pstSrcBlob, HI_RUNTIME_BLOB_S* pstDstBlob, HI_SAMPLE_SSD_PARAM_S* pstSsdParam) {
    HI_U32 i = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8* pu8VirAddr = NULL;
    HI_RUNTIME_MEM_S stMem;
    memset(&stMem, 0, sizeof(stMem));

    for(i = 0; i < 12; i++)
    {
        pstSsdParam->au32ConvHeight[i] = pstDstBlob[i].unShape.stWhc.u32Chn;
        pstSsdParam->au32ConvWidth[i] = pstDstBlob[i].unShape.stWhc.u32Height;
        pstSsdParam->au32ConvChannel[i] = pstDstBlob[i].unShape.stWhc.u32Width;
        if(i%2==1)
        {
            pstSsdParam->au32ConvStride[i/2] = ALIGN16(pstSsdParam->au32ConvChannel[i]*sizeof(HI_U32))/sizeof(HI_U32);
        }
    }

    /*Set PriorBox Parameters*/
    pstSsdParam->au32PriorBoxWidth[0] = 38;
    pstSsdParam->au32PriorBoxWidth[1] = 19;
    pstSsdParam->au32PriorBoxWidth[2] = 10;
    pstSsdParam->au32PriorBoxWidth[3] = 5;
    pstSsdParam->au32PriorBoxWidth[4] = 3;
    pstSsdParam->au32PriorBoxWidth[5] = 1;

    pstSsdParam->au32PriorBoxHeight[0] = 38;
    pstSsdParam->au32PriorBoxHeight[1] = 19;
    pstSsdParam->au32PriorBoxHeight[2] = 10;
    pstSsdParam->au32PriorBoxHeight[3] = 5;
    pstSsdParam->au32PriorBoxHeight[4] = 3;
    pstSsdParam->au32PriorBoxHeight[5] = 1;

    pstSsdParam->u32OriImHeight = 300;
    pstSsdParam->u32OriImWidth = 300;

    pstSsdParam->af32PriorBoxMinSize[0][0] = 30.0f;
    pstSsdParam->af32PriorBoxMinSize[1][0] = 60.0f;
    pstSsdParam->af32PriorBoxMinSize[2][0] = 111.0f;
    pstSsdParam->af32PriorBoxMinSize[3][0] = 162.0f;
    pstSsdParam->af32PriorBoxMinSize[4][0] = 213.0f;
    pstSsdParam->af32PriorBoxMinSize[5][0] = 264.0f;

    pstSsdParam->af32PriorBoxMaxSize[0][0] = 60.0f;
    pstSsdParam->af32PriorBoxMaxSize[1][0] = 111.0f;
    pstSsdParam->af32PriorBoxMaxSize[2][0] = 162.0f;
    pstSsdParam->af32PriorBoxMaxSize[3][0] = 213.0f;
    pstSsdParam->af32PriorBoxMaxSize[4][0] = 264.0f;
    pstSsdParam->af32PriorBoxMaxSize[5][0] = 315.0f;

    pstSsdParam->u32MinSizeNum = 1;
    pstSsdParam->u32MaxSizeNum = 1;
    pstSsdParam->bFlip= HI_TRUE;
    pstSsdParam->bClip= HI_FALSE;

    pstSsdParam->au32InputAspectRatioNum[0] = 1;
    pstSsdParam->au32InputAspectRatioNum[1] = 2;
    pstSsdParam->au32InputAspectRatioNum[2] = 2;
    pstSsdParam->au32InputAspectRatioNum[3] = 2;
    pstSsdParam->au32InputAspectRatioNum[4] = 1;
    pstSsdParam->au32InputAspectRatioNum[5] = 1;

    pstSsdParam->af32PriorBoxAspectRatio[0][0] = 2;
    pstSsdParam->af32PriorBoxAspectRatio[0][1] = 0;
    pstSsdParam->af32PriorBoxAspectRatio[1][0] = 2;
    pstSsdParam->af32PriorBoxAspectRatio[1][1] = 3;
    pstSsdParam->af32PriorBoxAspectRatio[2][0] = 2;
    pstSsdParam->af32PriorBoxAspectRatio[2][1] = 3;
    pstSsdParam->af32PriorBoxAspectRatio[3][0] = 2;
    pstSsdParam->af32PriorBoxAspectRatio[3][1] = 3;
    pstSsdParam->af32PriorBoxAspectRatio[4][0] = 2;
    pstSsdParam->af32PriorBoxAspectRatio[4][1] = 0;
    pstSsdParam->af32PriorBoxAspectRatio[5][0] = 2;
    pstSsdParam->af32PriorBoxAspectRatio[5][1] = 0;

    pstSsdParam->af32PriorBoxStepWidth[0] = 8;
    pstSsdParam->af32PriorBoxStepWidth[1] = 16;
    pstSsdParam->af32PriorBoxStepWidth[2] = 32;
    pstSsdParam->af32PriorBoxStepWidth[3] = 64;
    pstSsdParam->af32PriorBoxStepWidth[4] = 100;
    pstSsdParam->af32PriorBoxStepWidth[5] = 300;

    pstSsdParam->af32PriorBoxStepHeight[0] = 8;
    pstSsdParam->af32PriorBoxStepHeight[1] = 16;
    pstSsdParam->af32PriorBoxStepHeight[2] = 32;
    pstSsdParam->af32PriorBoxStepHeight[3] = 64;
    pstSsdParam->af32PriorBoxStepHeight[4] = 100;
    pstSsdParam->af32PriorBoxStepHeight[5] = 300;

    pstSsdParam->f32Offset = 0.5f;

    pstSsdParam->as32PriorBoxVar[0] = (HI_S32)(0.1f*SVP_WK_QUANT_BASE);
    pstSsdParam->as32PriorBoxVar[1] = (HI_S32)(0.1f*SVP_WK_QUANT_BASE);
    pstSsdParam->as32PriorBoxVar[2] = (HI_S32)(0.2f*SVP_WK_QUANT_BASE);
    pstSsdParam->as32PriorBoxVar[3] = (HI_S32)(0.2f*SVP_WK_QUANT_BASE);

    /*Set Softmax Parameters*/
    pstSsdParam->u32SoftMaxInHeight = 21;
    pstSsdParam->au32SoftMaxInChn[0] = 121296;
    pstSsdParam->au32SoftMaxInChn[1] = 45486;
    pstSsdParam->au32SoftMaxInChn[2] = 12600;
    pstSsdParam->au32SoftMaxInChn[3] = 3150;
    pstSsdParam->au32SoftMaxInChn[4] = 756;
    pstSsdParam->au32SoftMaxInChn[5] = 84;

    pstSsdParam->u32ConcatNum = 6;
    pstSsdParam->u32SoftMaxOutWidth = 1;
    pstSsdParam->u32SoftMaxOutHeight = 21;
    pstSsdParam->u32SoftMaxOutChn = 8732;

    /*Set DetectionOut Parameters*/
    pstSsdParam->u32ClassNum = 21;
    pstSsdParam->u32TopK = 400;
    pstSsdParam->u32KeepTopK = 200;
    pstSsdParam->u32NmsThresh = (HI_U32)(0.3f*SVP_WK_QUANT_BASE);
    pstSsdParam->u32ConfThresh = 1;
    pstSsdParam->au32DetectInputChn[0] = 23104;
    pstSsdParam->au32DetectInputChn[1] = 8664;
    pstSsdParam->au32DetectInputChn[2] = 2400;
    pstSsdParam->au32DetectInputChn[3] = 600;
    pstSsdParam->au32DetectInputChn[4] = 144;
    pstSsdParam->au32DetectInputChn[5] = 16;

    /*Malloc assist buffer memory*/
    u32ClassNum = pstSsdParam->u32ClassNum;
    u32TotalSize = SAMPLE_Ssd_GetResultTmpBuf(pstSsdParam);
    u32DstRoiSize = ALIGN16(u32ClassNum*pstSsdParam->u32TopK*sizeof(HI_U32)*SVP_WK_COORDI_NUM);
    u32DstScoreSize = ALIGN16(u32ClassNum*pstSsdParam->u32TopK*sizeof(HI_U32));
    u32ClassRoiNumSize = ALIGN16(u32ClassNum*sizeof(HI_U32));
    u32TotalSize = u32TotalSize+u32DstRoiSize+u32DstScoreSize+u32ClassRoiNumSize;

    stMem.u32Size = u32TotalSize;
    s32Ret = SAMPLE_AllocMem(&stMem, HI_TRUE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_Utils_AllocMem failed!\n");
    u64PhyAddr = stMem.u64PhyAddr;
    pu8VirAddr = (HI_U8*)((HI_UL)stMem.u64VirAddr);
    memset(pu8VirAddr, 0, u32TotalSize);
    s32Ret = SAMPLE_FlushCache(&stMem);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_Utils_FlushCache failed!\n");

   /*set each tmp buffer addr*/
    pstSsdParam->stPriorBoxTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSsdParam->stPriorBoxTmpBuf.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr);

    pstSsdParam->stSoftMaxTmpBuf.u64PhyAddr = u64PhyAddr+
        pstSsdParam->stPriorBoxTmpBuf.u32Size;
    pstSsdParam->stSoftMaxTmpBuf.u64VirAddr = (HI_U64)((HI_UL)(pu8VirAddr+
        pstSsdParam->stPriorBoxTmpBuf.u32Size));

    pstSsdParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr+
        pstSsdParam->stPriorBoxTmpBuf.u32Size+pstSsdParam->stSoftMaxTmpBuf.u32Size;
    pstSsdParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)((HI_UL)(pu8VirAddr+
        pstSsdParam->stPriorBoxTmpBuf.u32Size+ pstSsdParam->stSoftMaxTmpBuf.u32Size));

    u32TmpBufTotalSize = pstSsdParam->stPriorBoxTmpBuf.u32Size+
        pstSsdParam->stSoftMaxTmpBuf.u32Size + pstSsdParam->stGetResultTmpBuf.u32Size;

    /*set result blob*/
    pstSsdParam->stDstRoi.u64PhyAddr = u64PhyAddr+u32TmpBufTotalSize;
    pstSsdParam->stDstRoi.u64VirAddr = (HI_U64)((HI_UL)(pu8VirAddr+u32TmpBufTotalSize));

    pstSsdParam->stDstScore.u64PhyAddr = u64PhyAddr+u32TmpBufTotalSize+u32DstRoiSize;
    pstSsdParam->stDstScore.u64VirAddr = (HI_U64)((HI_UL)(pu8VirAddr+u32TmpBufTotalSize+u32DstRoiSize));

    pstSsdParam->stClassRoiNum.u64PhyAddr = u64PhyAddr+u32TmpBufTotalSize+
        u32DstRoiSize+u32DstScoreSize;
    pstSsdParam->stClassRoiNum.u64VirAddr = (HI_U64)((HI_UL)(pu8VirAddr+u32TmpBufTotalSize+
        u32DstRoiSize+u32DstScoreSize));

    return s32Ret;
FAIL_0:
    SAMPLE_FreeMem(&stMem);
    return s32Ret;
}

static HI_S32 SAMPLE_Ssd_PriorBoxForward(HI_U32 u32PriorBoxWidth,
    HI_U32 u32PriorBoxHeight, HI_U32 u32OriImWidth, HI_U32 u32OriImHeight,
    HI_FLOAT* pf32PriorBoxMinSize, HI_U32 u32MinSizeNum, HI_FLOAT* pf32PriorBoxMaxSize,
    HI_U32 u32MaxSizeNum, HI_BOOL bFlip, HI_BOOL bClip, HI_U32 u32InputAspectRatioNum,
    HI_FLOAT af32PriorBoxAspectRatio[],HI_FLOAT f32PriorBoxStepWidth,
    HI_FLOAT f32PriorBoxStepHeight,HI_FLOAT f32Offset,HI_S32 as32PriorBoxVar[],
    HI_S32* ps32PriorboxOutputData)
{
    HI_U32 u32AspectRatioNum = 0;
    HI_U32 u32Index = 0;
    HI_FLOAT af32AspectRatio[SAMPLE_SSD_ASPECT_RATIO_NUM] = { 0 };
    HI_U32 u32NumPrior = 0;
    HI_FLOAT f32CenterX = 0;
    HI_FLOAT f32CenterY = 0;
    HI_FLOAT f32BoxHeight = 0;
    HI_FLOAT f32BoxWidth = 0;
    HI_FLOAT f32MaxBoxWidth = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 n = 0;
    HI_U32 h = 0;
    HI_U32 w = 0;
    SAMPLE_CHK_RETURN((HI_TRUE == bFlip && u32InputAspectRatioNum >
        (SAMPLE_SSD_ASPECT_RATIO_NUM-1)/2), HI_FAILURE,
        "Error,when bFlip is true, u32InputAspectRatioNum(%d) can't be greater than %d!\n",
        u32InputAspectRatioNum, (SAMPLE_SSD_ASPECT_RATIO_NUM-1)/2);
    SAMPLE_CHK_RETURN((HI_FALSE == bFlip && u32InputAspectRatioNum >
        (SAMPLE_SSD_ASPECT_RATIO_NUM-1)), HI_FAILURE,
        "Error,when bFlip is false, u32InputAspectRatioNum(%d) can't be greater than %d!\n",
        u32InputAspectRatioNum, (SAMPLE_SSD_ASPECT_RATIO_NUM-1));

    // generate aspect_ratios
    u32AspectRatioNum = 0;
    af32AspectRatio[0] = 1;
    u32AspectRatioNum++;
    for (i = 0; i < u32InputAspectRatioNum; i++)
    {
        af32AspectRatio[u32AspectRatioNum++] = af32PriorBoxAspectRatio[i];
        if (bFlip)
        {
            af32AspectRatio[u32AspectRatioNum++] = 1.0f / af32PriorBoxAspectRatio[i];
        }
    }
    u32NumPrior = u32MinSizeNum * u32AspectRatioNum + u32MaxSizeNum;

    u32Index = 0;
    for (h = 0; h < u32PriorBoxHeight; h++)
    {
        for (w = 0; w < u32PriorBoxWidth; w++)
        {
            f32CenterX = (w + f32Offset) * f32PriorBoxStepWidth;
            f32CenterY = (h + f32Offset) * f32PriorBoxStepHeight;
            for (n = 0; n < u32MinSizeNum; n++)
            {
                /*** first prior ***/
                f32BoxHeight = pf32PriorBoxMinSize[n];
                f32BoxWidth = pf32PriorBoxMinSize[n];
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                /*** second prior ***/
                if(u32MaxSizeNum>0)
                {
                    f32MaxBoxWidth = sqrt(pf32PriorBoxMinSize[n] * pf32PriorBoxMaxSize[n]);
                    f32BoxHeight = f32MaxBoxWidth;
                    f32BoxWidth = f32MaxBoxWidth;
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                }
                /**** rest of priors, skip AspectRatio == 1 ****/
                for (i = 1; i < u32AspectRatioNum; i++)
                {
                    f32BoxWidth = (HI_FLOAT)(pf32PriorBoxMinSize[n] * sqrt( af32AspectRatio[i] ));
                    f32BoxHeight = (HI_FLOAT)(pf32PriorBoxMinSize[n]/sqrt( af32AspectRatio[i] ));
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX - f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY - f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterX + f32BoxWidth * SAMPLE_SVP_NNIE_HALF);
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)(f32CenterY + f32BoxHeight * SAMPLE_SVP_NNIE_HALF);
                }
            }
        }
    }
    /************ clip the priors' coordidates, within [0, u32ImgWidth] & [0, u32ImgHeight] *************/
    if (bClip)
    {
        for (i = 0; i < (HI_U32)(u32PriorBoxWidth * u32PriorBoxHeight * SVP_WK_COORDI_NUM*u32NumPrior / 2); i++)
        {
            ps32PriorboxOutputData[2 * i] = min((HI_U32)max(ps32PriorboxOutputData[2 * i], 0), u32OriImWidth);
            ps32PriorboxOutputData[2 * i + 1] = min((HI_U32)max(ps32PriorboxOutputData[2 * i + 1], 0), u32OriImHeight);
        }
    }
    /*********************** get var **********************/
    for (h = 0; h < u32PriorBoxHeight; h++)
    {
        for (w = 0; w < u32PriorBoxWidth; w++)
        {
            for (i = 0; i < u32NumPrior; i++)
            {
                for (j = 0; j < SVP_WK_COORDI_NUM; j++)
                {
                    ps32PriorboxOutputData[u32Index++] = (HI_S32)as32PriorBoxVar[j];
                }
            }
        }
    }
    return HI_SUCCESS;
}

static HI_S32 SAMPLE_SSD_SoftMax(HI_S32* ps32Src, HI_S32 s32ArraySize, HI_S32* ps32Dst)
{
    /***** define parameters ****/
    HI_S32 s32Max = 0;
    HI_S32 s32Sum = 0;
    HI_S32 i = 0;
    for (i = 0; i < s32ArraySize; ++i)
    {
        if (s32Max < ps32Src[i])
        {
            s32Max = ps32Src[i];
        }
    }
    for (i = 0; i < s32ArraySize; ++i)
    {
        ps32Dst[i] = (HI_S32)(SVP_WK_QUANT_BASE* exp((HI_FLOAT)(ps32Src[i] - s32Max) / SVP_WK_QUANT_BASE));
        s32Sum += ps32Dst[i];
    }
    for (i = 0; i < s32ArraySize; ++i)
    {
        ps32Dst[i] = (HI_S32)(((HI_FLOAT)ps32Dst[i] / (HI_FLOAT)s32Sum) * SVP_WK_QUANT_BASE);
    }
    return HI_SUCCESS;
}

static HI_S32 SAMPLE_Ssd_SoftmaxForward(HI_U32 u32SoftMaxInHeight,
    HI_U32 au32SoftMaxInChn[], HI_U32 u32ConcatNum, HI_U32 au32ConvStride[],
    HI_U32 u32SoftMaxOutWidth, HI_U32 u32SoftMaxOutHeight, HI_U32 u32SoftMaxOutChn,
    HI_S32* aps32SoftMaxInputData[], HI_S32* ps32SoftMaxOutputData)
{
    HI_S32* ps32InputData = NULL;
    HI_S32* ps32OutputTmp = NULL;
    HI_U32 u32OuterNum = 0;
    HI_U32 u32InnerNum = 0;
    HI_U32 u32InputChannel = 0;
    HI_U32 i = 0;
    HI_U32 u32ConcatCnt = 0;
    HI_S32 s32Ret = 0;
    HI_U32 u32Stride = 0;
    HI_U32 u32Skip = 0;
    HI_U32 u32Left = 0;
    ps32OutputTmp = ps32SoftMaxOutputData;
    for (u32ConcatCnt = 0; u32ConcatCnt < u32ConcatNum; u32ConcatCnt++)
    {
        ps32InputData = aps32SoftMaxInputData[u32ConcatCnt];
        u32Stride = au32ConvStride[u32ConcatCnt];
        u32InputChannel = au32SoftMaxInChn[u32ConcatCnt];
        u32OuterNum = u32InputChannel / u32SoftMaxInHeight;
        u32InnerNum = u32SoftMaxInHeight;
        u32Skip = u32Stride / u32InnerNum;
        u32Left = u32Stride % u32InnerNum;        // do softmax
        for (i = 0; i < u32OuterNum; i++)
        {
            s32Ret = SAMPLE_SSD_SoftMax(ps32InputData, (HI_S32)u32InnerNum,ps32OutputTmp);
            if ((i + 1) % u32Skip == 0)
            {
                ps32InputData += u32Left;
            }
            ps32InputData += u32InnerNum;
            ps32OutputTmp += u32InnerNum;
        }
    }
    return s32Ret;
}


static HI_S32 SAMPLE_Ssd_DetectionOutForward(HI_U32 u32ConcatNum,
    HI_U32 u32ConfThresh,HI_U32 u32ClassNum, HI_U32 u32TopK, HI_U32 u32KeepTopK, HI_U32 u32NmsThresh,
    HI_U32 au32DetectInputChn[], HI_S32* aps32AllLocPreds[], HI_S32* aps32AllPriorBoxes[],
    HI_S32* ps32ConfScores, HI_S32* ps32AssistMemPool, HI_S32* ps32DstScoreSrc,
    HI_S32* ps32DstBboxSrc, HI_S32* ps32RoiOutCntSrc)
{
    /************* check input parameters ****************/
    /******** define variables **********/
    HI_S32* ps32LocPreds = NULL;
    HI_S32* ps32PriorBoxes = NULL;
    HI_S32* ps32PriorVar = NULL;
    HI_S32* ps32AllDecodeBoxes = NULL;
    HI_S32* ps32DstScore = NULL;
    HI_S32* ps32DstBbox = NULL;
    HI_S32* ps32ClassRoiNum = NULL;
    HI_U32 u32RoiOutCnt = 0;
    HI_S32* ps32SingleProposal = NULL;
    HI_S32* ps32AfterTopK = NULL;
    NNIE_STACK_S* pstStack = NULL;
    HI_U32 u32PriorNum = 0;
    HI_U32 u32NumPredsPerClass = 0;
    HI_FLOAT f32PriorWidth = 0;
    HI_FLOAT f32PriorHeight = 0;
    HI_FLOAT f32PriorCenterX = 0;
    HI_FLOAT f32PriorCenterY = 0;
    HI_FLOAT f32DecodeBoxCenterX = 0;
    HI_FLOAT f32DecodeBoxCenterY = 0;
    HI_FLOAT f32DecodeBoxWidth = 0;
    HI_FLOAT f32DecodeBoxHeight = 0;
    HI_U32 u32SrcIdx = 0;
    HI_U32 u32AfterFilter = 0;
    HI_U32 u32AfterTopK = 0;
    HI_U32 u32KeepCnt = 0;
    HI_U32 i = 0;
    HI_U32 j = 0;
    HI_U32 u32Offset = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    u32PriorNum = 0;
    for (i = 0; i < u32ConcatNum; i++)
    {
        u32PriorNum += au32DetectInputChn[i] / SVP_WK_COORDI_NUM;
    }
    //prepare for Assist MemPool
    ps32AllDecodeBoxes = ps32AssistMemPool;
    ps32SingleProposal = ps32AllDecodeBoxes + u32PriorNum * SVP_WK_COORDI_NUM;
    ps32AfterTopK = ps32SingleProposal + SVP_WK_PROPOSAL_WIDTH * u32PriorNum;
    pstStack = (NNIE_STACK_S*)(ps32AfterTopK + u32PriorNum * SVP_WK_PROPOSAL_WIDTH);
    u32SrcIdx = 0;
    for (i = 0; i < u32ConcatNum; i++)
    {
        /********** get loc predictions ************/
        ps32LocPreds = aps32AllLocPreds[i];
        u32NumPredsPerClass = au32DetectInputChn[i] / SVP_WK_COORDI_NUM;
        /********** get Prior Bboxes ************/
        ps32PriorBoxes = aps32AllPriorBoxes[i];
        ps32PriorVar = ps32PriorBoxes + u32NumPredsPerClass*SVP_WK_COORDI_NUM;
        for (j = 0; j < u32NumPredsPerClass; j++)
        {
            //printf("ps32PriorBoxes start***************\n");
            f32PriorWidth = (HI_FLOAT)(ps32PriorBoxes[j*SVP_WK_COORDI_NUM+2] - ps32PriorBoxes[j*SVP_WK_COORDI_NUM]);
            f32PriorHeight = (HI_FLOAT)(ps32PriorBoxes[j*SVP_WK_COORDI_NUM+3] - ps32PriorBoxes[j*SVP_WK_COORDI_NUM + 1]);
            f32PriorCenterX = (ps32PriorBoxes[j*SVP_WK_COORDI_NUM+2] + ps32PriorBoxes[j*SVP_WK_COORDI_NUM])*SAMPLE_SVP_NNIE_HALF;
            f32PriorCenterY = (ps32PriorBoxes[j*SVP_WK_COORDI_NUM+3] + ps32PriorBoxes[j*SVP_WK_COORDI_NUM+1])*SAMPLE_SVP_NNIE_HALF;

            f32DecodeBoxCenterX = ((HI_FLOAT)ps32PriorVar[j*SVP_WK_COORDI_NUM]/SVP_WK_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SVP_WK_COORDI_NUM]/SVP_WK_QUANT_BASE)*f32PriorWidth+f32PriorCenterX;

            f32DecodeBoxCenterY = ((HI_FLOAT)ps32PriorVar[j*SVP_WK_COORDI_NUM+1]/SVP_WK_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SVP_WK_COORDI_NUM+1]/SVP_WK_QUANT_BASE)*f32PriorHeight+f32PriorCenterY;

            f32DecodeBoxWidth = exp(((HI_FLOAT)ps32PriorVar[j*SVP_WK_COORDI_NUM+2]/SVP_WK_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SVP_WK_COORDI_NUM+2]/SVP_WK_QUANT_BASE))*f32PriorWidth;

            f32DecodeBoxHeight = exp(((HI_FLOAT)ps32PriorVar[j*SVP_WK_COORDI_NUM+3]/SVP_WK_QUANT_BASE)*
                ((HI_FLOAT)ps32LocPreds[j*SVP_WK_COORDI_NUM+3]/SVP_WK_QUANT_BASE))*f32PriorHeight;

            //printf("ps32PriorBoxes end***************\n");

            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX - f32DecodeBoxWidth * SAMPLE_SVP_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY - f32DecodeBoxHeight * SAMPLE_SVP_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterX + f32DecodeBoxWidth * SAMPLE_SVP_NNIE_HALF);
            ps32AllDecodeBoxes[u32SrcIdx++] = (HI_S32)(f32DecodeBoxCenterY + f32DecodeBoxHeight * SAMPLE_SVP_NNIE_HALF);
        }
    }
    // do NMS for each class
    u32AfterTopK = 0;
    for (i = 0; i < u32ClassNum; i++)
    {
        for (j = 0; j < u32PriorNum; j++)
        {
            ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH] = ps32AllDecodeBoxes[j * SVP_WK_COORDI_NUM];
            ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 1] = ps32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 1];
            ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 2] = ps32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 2];
            ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 3] = ps32AllDecodeBoxes[j * SVP_WK_COORDI_NUM + 3];
            ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 4] = ps32ConfScores[j*u32ClassNum + i];
            ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 5] = 0;
        }
        s32Ret = NonRecursiveArgQuickSort(ps32SingleProposal, 0, u32PriorNum - 1, pstStack, u32TopK);
        u32AfterFilter = (u32PriorNum < u32TopK) ? u32PriorNum : u32TopK;
        s32Ret = NonMaxSuppression(ps32SingleProposal, u32AfterFilter, u32NmsThresh);
        u32RoiOutCnt = 0;
        ps32DstScore = (HI_S32*)ps32DstScoreSrc;
        ps32DstBbox = (HI_S32*)ps32DstBboxSrc;
        ps32ClassRoiNum = (HI_S32*)ps32RoiOutCntSrc;
        ps32DstScore += (HI_S32)u32AfterTopK;
        ps32DstBbox += (HI_S32)(u32AfterTopK * SVP_WK_COORDI_NUM);
        for (j = 0; j < u32TopK; j++)
        {
            if (ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 5] == 0 &&
                ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 4] > (HI_S32)u32NmsThresh)
            {
                ps32DstScore[u32RoiOutCnt] = ps32SingleProposal[j * 6 + 4];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM] = ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1] = ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 1];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2] = ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 2];
                ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3] = ps32SingleProposal[j * SVP_WK_PROPOSAL_WIDTH + 3];
                u32RoiOutCnt++;
            }
        }
        ps32ClassRoiNum[i] = (HI_S32)u32RoiOutCnt;
        u32AfterTopK += u32RoiOutCnt;
    }

    u32KeepCnt = 0;
    u32Offset = 0;
    if (u32AfterTopK > u32KeepTopK)
    {
        SAMPLE_CHK_RETURN(HI_NULL == ps32ClassRoiNum, HI_FAILURE, "ps32ClassRoiNum is null");
        u32Offset = ps32ClassRoiNum[0];
        for (i = 1; i < u32ClassNum; i++)
        {
            ps32DstScore = (HI_S32*)ps32DstScoreSrc;
            ps32DstBbox = (HI_S32*)ps32DstBboxSrc;
            ps32ClassRoiNum = (HI_S32*)ps32RoiOutCntSrc;
            ps32DstScore += (HI_S32)(u32Offset);
            ps32DstBbox += (HI_S32)(u32Offset * SVP_WK_COORDI_NUM);
            for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
            {
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH] = ps32DstBbox[j * SVP_WK_COORDI_NUM];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 1] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 1];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 2] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 2];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 3] = ps32DstBbox[j * SVP_WK_COORDI_NUM + 3];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 4] = ps32DstScore[j];
                ps32AfterTopK[u32KeepCnt * SVP_WK_PROPOSAL_WIDTH + 5] = i;
                u32KeepCnt++;
            }
            u32Offset = u32Offset + ps32ClassRoiNum[i];
        }
        s32Ret = NonRecursiveArgQuickSort(ps32AfterTopK, 0, u32KeepCnt - 1, pstStack, u32KeepCnt);

        u32Offset = 0;
        u32Offset = ps32ClassRoiNum[0];
        for (i = 1; i < u32ClassNum; i++)
        {
            u32RoiOutCnt = 0;
            ps32DstScore = (HI_S32*)ps32DstScoreSrc;
            ps32DstBbox = (HI_S32*)ps32DstBboxSrc;
            ps32ClassRoiNum = (HI_S32*)ps32RoiOutCntSrc;
            ps32DstScore += (HI_S32)(u32Offset);
            ps32DstBbox += (HI_S32)(u32Offset * SVP_WK_COORDI_NUM);
            for (j = 0; j < u32KeepTopK; j++)
            {
                if (ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 5] == i)
                {
                    ps32DstScore[u32RoiOutCnt] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 4];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 1] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 1];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 2] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 2];
                    ps32DstBbox[u32RoiOutCnt * SVP_WK_COORDI_NUM + 3] = ps32AfterTopK[j * SVP_WK_PROPOSAL_WIDTH + 3];
                    u32RoiOutCnt++;
                }
            }
            ps32ClassRoiNum[i] = (HI_S32)u32RoiOutCnt;
            u32Offset += u32RoiOutCnt;
        }
    }
    return s32Ret;
}

static HI_S32 SAMPLE_SSD_Detection_PrintResult(HI_RUNTIME_MEM_S *pstDstScore,
                    HI_RUNTIME_MEM_S *pstDstRoi, HI_RUNTIME_MEM_S *pstClassRoiNum,
                    HI_FLOAT f32PrintResultThresh, HI_S32* ps32ResultROI, HI_U32* pu32ResultROICnt)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32* ps32Score = (HI_S32*)((HI_UL)pstDstScore->u64VirAddr);
    HI_S32* ps32Roi = (HI_S32*)((HI_UL)pstDstRoi->u64VirAddr);
    HI_S32* ps32ClassRoiNum = (HI_S32*)((HI_UL)pstClassRoiNum->u64VirAddr);
    HI_U32 u32ClassNum = 21;
    HI_S32 s32XMin = 0,s32YMin= 0,s32XMax = 0,s32YMax = 0;

    u32RoiNumBias += ps32ClassRoiNum[0];
    for (i = 1; i < u32ClassNum; i++)
    {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * SVP_WK_COORDI_NUM;
        // if the confidence score greater than result threshold, the result will be printed
        if((HI_FLOAT)ps32Score[u32ScoreBias] / SVP_WK_QUANT_BASE >=
            f32PrintResultThresh && ps32ClassRoiNum[i]!=0)
        {
            printf("==== The %dth class box info====\n", i);
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++)
        {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / SVP_WK_QUANT_BASE;
            if (f32Score < f32PrintResultThresh)
            {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j*SVP_WK_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j*SVP_WK_COORDI_NUM + 1];
            s32XMax = ps32Roi[u32BboxBias + j*SVP_WK_COORDI_NUM + 2];
            s32YMax = ps32Roi[u32BboxBias + j*SVP_WK_COORDI_NUM + 3];
            printf("%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);

            memcpy(&ps32ResultROI[*pu32ResultROICnt * SVP_WK_PROPOSAL_WIDTH],
                   &ps32Roi[u32BboxBias + j * SVP_WK_COORDI_NUM], sizeof(HI_S32)*SVP_WK_COORDI_NUM);
            ps32ResultROI[*pu32ResultROICnt * SVP_WK_PROPOSAL_WIDTH + 4] = ps32Score[u32ScoreBias + j];
            ps32ResultROI[*pu32ResultROICnt * SVP_WK_PROPOSAL_WIDTH + 5] = RPN_SUPPRESS_FALSE;
            *pu32ResultROICnt = *pu32ResultROICnt + 1;
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    return HI_SUCCESS;
}

HI_S32 SAMPLE_Ssd_GetResult(HI_RUNTIME_BLOB_S* pstSrcBlob, HI_RUNTIME_BLOB_S* pstDstBlob,
                        HI_S32* ps32ResultROI, HI_U32* pu32ResultROICnt) {
    HI_SAMPLE_SSD_PARAM_S ssdParam;
    HI_S32* aps32PermuteResult[SAMPLE_SSD_REPORT_NODE_NUM];
    HI_S32* aps32PriorboxOutputData[SAMPLE_SSD_PRIORBOX_NUM];
    HI_S32* aps32SoftMaxInputData[SAMPLE_SSD_SOFTMAX_NUM];
    HI_S32* aps32DetectionLocData[SAMPLE_SSD_SOFTMAX_NUM];
    HI_S32* ps32SoftMaxOutputData = NULL;
    HI_S32* ps32DetectionOutTmpBuf = NULL;
    HI_U32 u32Size = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0;
    HI_FLOAT f32PrintResultThresh = 0.8f;

    memset(&ssdParam, 0, sizeof(ssdParam));

    s32Ret = SAMPLE_Ssd_InitParam(pstSrcBlob, pstDstBlob, &ssdParam);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "SAMPLE_Ssd_InitParam failed!\n");
    /*get permut result*/
    for(i = 0; i < SAMPLE_SSD_REPORT_NODE_NUM; i++)
    {
        aps32PermuteResult[i] = (HI_S32*)((HI_UL)pstDstBlob[i].u64VirAddr);
    }

    /*priorbox*/
    aps32PriorboxOutputData[0] = (HI_S32*)((HI_UL)ssdParam.stPriorBoxTmpBuf.u64VirAddr);
    for (i = 1; i < SAMPLE_SSD_PRIORBOX_NUM; i++)
    {
        u32Size = ssdParam.au32PriorBoxHeight[i-1]*ssdParam.au32PriorBoxWidth[i-1]*
            SVP_WK_COORDI_NUM*2*(ssdParam.u32MaxSizeNum+ssdParam.u32MinSizeNum+
            ssdParam.au32InputAspectRatioNum[i-1]*2*ssdParam.u32MinSizeNum);
        aps32PriorboxOutputData[i] = aps32PriorboxOutputData[i - 1] + u32Size;
    }

    for (i = 0; i < SAMPLE_SSD_PRIORBOX_NUM; i++)
    {
        s32Ret = SAMPLE_Ssd_PriorBoxForward(ssdParam.au32PriorBoxWidth[i],
            ssdParam.au32PriorBoxHeight[i], ssdParam.u32OriImWidth,
            ssdParam.u32OriImHeight, ssdParam.af32PriorBoxMinSize[i],
            ssdParam.u32MinSizeNum,ssdParam.af32PriorBoxMaxSize[i],
            ssdParam.u32MaxSizeNum, ssdParam.bFlip, ssdParam.bClip,
            ssdParam.au32InputAspectRatioNum[i],ssdParam.af32PriorBoxAspectRatio[i],
            ssdParam.af32PriorBoxStepWidth[i],ssdParam.af32PriorBoxStepHeight[i],
            ssdParam.f32Offset,ssdParam.as32PriorBoxVar,
            aps32PriorboxOutputData[i]);
        SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0,
                                   "Error,SAMPLE_Ssd_PriorBoxForward failed!\n");
    }

    /*softmax*/
    ps32SoftMaxOutputData = (HI_S32*)((HI_UL)ssdParam.stSoftMaxTmpBuf.u64VirAddr);
    for(i = 0; i < SAMPLE_SSD_SOFTMAX_NUM; i++)
    {
        aps32SoftMaxInputData[i] = aps32PermuteResult[i*2+1];
    }

    (void)SAMPLE_Ssd_SoftmaxForward(ssdParam.u32SoftMaxInHeight,
        ssdParam.au32SoftMaxInChn, ssdParam.u32ConcatNum,
        ssdParam.au32ConvStride, ssdParam.u32SoftMaxOutWidth,
        ssdParam.u32SoftMaxOutHeight, ssdParam.u32SoftMaxOutChn,
        aps32SoftMaxInputData, ps32SoftMaxOutputData);

    /*detection*/
    ps32DetectionOutTmpBuf = (HI_S32*)((HI_UL)ssdParam.stGetResultTmpBuf.u64VirAddr);
    for(i = 0; i < SAMPLE_SSD_PRIORBOX_NUM; i++)
    {
        aps32DetectionLocData[i] = aps32PermuteResult[i*2];
    }

    (void)SAMPLE_Ssd_DetectionOutForward(ssdParam.u32ConcatNum,
        ssdParam.u32ConfThresh,ssdParam.u32ClassNum, ssdParam.u32TopK,
        ssdParam.u32KeepTopK, ssdParam.u32NmsThresh,ssdParam.au32DetectInputChn,
        aps32DetectionLocData, aps32PriorboxOutputData, ps32SoftMaxOutputData,
        ps32DetectionOutTmpBuf, (HI_S32*)((HI_UL)ssdParam.stDstScore.u64VirAddr),
        (HI_S32*)((HI_UL)ssdParam.stDstRoi.u64VirAddr),
        (HI_S32*)((HI_UL)ssdParam.stClassRoiNum.u64VirAddr));

    (void)SAMPLE_SSD_Detection_PrintResult(&ssdParam.stDstScore,
        &ssdParam.stDstRoi, &ssdParam.stClassRoiNum,f32PrintResultThresh, ps32ResultROI, pu32ResultROICnt);

FAIL_0:
    SAMPLE_FreeMem(&ssdParam.stPriorBoxTmpBuf);
    return s32Ret;
}

HI_S32 SAMPLE_DATA_GetRoiResultFromOriginSize(SAMPLE_RUNTIME_MODEL_TYPE_E enType,
        HI_RUNTIME_BLOB_S* pstScoreBlob,
        HI_RUNTIME_BLOB_S* pstBBoxBlob,
        HI_RUNTIME_BLOB_S* pstProposalBlob,
        HI_U32 u32Width,
        HI_U32 u32Height,
        HI_S32* ps32ResultROI,
        HI_U32* pu32ResultROICnt)
{
    HI_U32 i, j;
    HI_U32 u32RoiCnt = pstProposalBlob->unShape.stWhc.u32Height;
    HI_U32 u32ScoreStride = pstScoreBlob->u32Stride;
    HI_U32 u32BBoxStride = ALIGN32(pstBBoxBlob->unShape.stWhc.u32Width * sizeof(HI_S32));
    HI_S32* ps32Score = (HI_S32*)((HI_UL)pstScoreBlob->u64VirAddr);
    HI_U32 u32ClassNum = pstScoreBlob->unShape.stWhc.u32Width;
    HI_S32* ps32Proposal = (HI_S32*)((HI_UL)pstProposalBlob->u64VirAddr);
    HI_S32* ps32BBox = (HI_S32*)((HI_UL)pstBBoxBlob->u64VirAddr);
    HI_S32* ps32AllBoxes = HI_NULL;
    HI_S32* ps32AllBoxesTmp = HI_NULL;
    HI_FLOAT fProposalXMin, fProposalXMax, fProposalYMin, fProposalYMax, fProposalWidth, fProposalHeight, fProposalCenterX, fProposalCenterY;
    HI_FLOAT fBBoxCenterXDelta, fBBoxCenterYDelta, fBBoxWidthDelta, fBBoxHeightDelta;
    HI_FLOAT fPredWidth, fPredHeight, fPredCenterX, fPredCenterY;
    NNIE_STACK_S* pstStack = HI_NULL;
    HI_U32 u32ResultBoxNum = 0;

    SAMPLE_CHK_GOTO(0 == u32RoiCnt, FAIL_0, "u32RoiCnt equals 0\n");

    ps32AllBoxes = (HI_S32*)malloc(u32RoiCnt * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_S32));
    SAMPLE_CHK_GOTO(HI_NULL == ps32AllBoxes, FAIL_0, "u32RoiCnt equals 0\n");

    pstStack = (NNIE_STACK_S*)malloc(sizeof(NNIE_STACK_S) * u32RoiCnt);
    SAMPLE_CHK_GOTO(HI_NULL == pstStack, FAIL_0, "u32RoiCnt equals 0\n");

    for (j = 0; j < u32ClassNum; j++)
    {
        for (i = 0; i < u32RoiCnt; i++)
        {
            fProposalXMin = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM] / 4096.0);
            fProposalYMin = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM + 1] / 4096.0);
            fProposalXMax = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM + 2] / 4096.0);
            fProposalYMax = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM + 3] / 4096.0);

            if (SAMPLE_RUNTIME_MODEL_TYPE_RFCN == enType)
            {
                fBBoxCenterXDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM] / 4096.0);
                fBBoxCenterYDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM + 1] / 4096.0);
                fBBoxWidthDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM + 2] / 4096.0);
                fBBoxHeightDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM + 3] / 4096.0);
            }
            else
            {
                fBBoxCenterXDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM] / 4096.0);
                fBBoxCenterYDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM + 1] / 4096.0);
                fBBoxWidthDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM + 2] / 4096.0);
                fBBoxHeightDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM + 3] / 4096.0);
            }

            fProposalWidth = fProposalXMax - fProposalXMin + 1;
            fProposalHeight = fProposalYMax - fProposalYMin + 1;
            fProposalCenterX = (HI_FLOAT)(fProposalXMin + fProposalWidth * 0.5);
            fProposalCenterY = (HI_FLOAT)(fProposalYMin + fProposalHeight * 0.5);
            fPredWidth = (HI_FLOAT)(fProposalWidth * exp(fBBoxWidthDelta));
            fPredHeight = (HI_FLOAT)(fProposalHeight * exp(fBBoxHeightDelta));
            fPredCenterX = fProposalCenterX + fProposalWidth * fBBoxCenterXDelta;
            fPredCenterY = fProposalCenterY + fProposalHeight * fBBoxCenterYDelta;
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH] = (HI_S32)(fPredCenterX - 0.5 * fPredWidth);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 1] = (HI_S32)(fPredCenterY - 0.5 * fPredHeight);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 2] = (HI_S32)(fPredCenterX + 0.5 * fPredWidth);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 3] = (HI_S32)(fPredCenterY + 0.5 * fPredHeight);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 4] = ps32Score[i * u32ScoreStride / sizeof(HI_S32) + j];
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 5] = RPN_SUPPRESS_FALSE; // RPN Suppress
        }

        BboxClip_N(ps32AllBoxes, u32Width, u32Height, u32RoiCnt);
        ps32AllBoxesTmp = ps32AllBoxes;
        NonRecursiveArgQuickSort(ps32AllBoxesTmp, 0, u32RoiCnt - 1, pstStack, u32RoiCnt);
        NonMaxSuppression(ps32AllBoxesTmp, u32RoiCnt, (HI_U32)(0.7 * SVP_WK_QUANT_BASE));
        u32ResultBoxNum = 0;

        for (i = 0; i < u32RoiCnt; i++)
        {
            if (RPN_SUPPRESS_FALSE == ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 5]
                && ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 4] > 0.8 * SVP_WK_QUANT_BASE)
            {
#if SAMPLE_DEBUG
                printf("%d %f %d %d %d %d\n", j, ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 4] / 4096.0,
                       ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 0], ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 1],
                       ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 2], ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 3]);
#endif
                u32ResultBoxNum++;

                if (0 != j) // not background
                {
                    memcpy(&ps32ResultROI[*pu32ResultROICnt * SVP_WK_PROPOSAL_WIDTH], 
                           &ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH], sizeof(HI_S32)*SVP_WK_PROPOSAL_WIDTH);
                    *pu32ResultROICnt = *pu32ResultROICnt + 1;
                }
            }
        }

#if SAMPLE_DEBUG
        printf("class %d has %d boxes\n", j, u32ResultBoxNum);
#endif
    }

    SAMPLE_FREE(pstStack);
    SAMPLE_FREE(ps32AllBoxes);
    return HI_SUCCESS;
FAIL_0:
    SAMPLE_FREE(pstStack);
    SAMPLE_FREE(ps32AllBoxes);
    return HI_FAILURE;
}

HI_S32 SAMPLE_DATA_GetRoiResult(SAMPLE_RUNTIME_MODEL_TYPE_E enType,
        HI_RUNTIME_BLOB_S* pstScoreBlob,
        HI_RUNTIME_BLOB_S* pstBBoxBlob,
        HI_RUNTIME_BLOB_S* pstProposalBlob,
        HI_RUNTIME_BLOB_S* pstDataBlob,
        HI_S32* ps32ResultROI,
        HI_U32* pu32ResultROICnt)
{
    HI_U32 i, j;
    HI_U32 u32RoiCnt = pstProposalBlob->unShape.stWhc.u32Height;
    HI_U32 u32ScoreStride = pstScoreBlob->u32Stride;
    HI_U32 u32BBoxStride = ALIGN32(pstBBoxBlob->unShape.stWhc.u32Width * sizeof(HI_S32));
    HI_S32* ps32Score = (HI_S32*)((HI_UL)pstScoreBlob->u64VirAddr);
    HI_U32 u32ClassNum = pstScoreBlob->unShape.stWhc.u32Width;
    HI_S32* ps32Proposal = (HI_S32*)((HI_UL)pstProposalBlob->u64VirAddr);
    HI_S32* ps32BBox = (HI_S32*)((HI_UL)pstBBoxBlob->u64VirAddr);
    HI_S32* ps32AllBoxes = HI_NULL;
    HI_S32* ps32AllBoxesTmp = HI_NULL;
    HI_FLOAT fProposalXMin, fProposalXMax, fProposalYMin, fProposalYMax, fProposalWidth, fProposalHeight, fProposalCenterX, fProposalCenterY;
    HI_FLOAT fBBoxCenterXDelta, fBBoxCenterYDelta, fBBoxWidthDelta, fBBoxHeightDelta;
    HI_FLOAT fPredWidth, fPredHeight, fPredCenterX, fPredCenterY;
    NNIE_STACK_S* pstStack = HI_NULL;
    HI_U32 u32ResultBoxNum = 0;

    SAMPLE_CHK_GOTO(0 == u32RoiCnt, FAIL_0, "u32RoiCnt equals 0\n");

    ps32AllBoxes = (HI_S32*)malloc(u32RoiCnt * SVP_WK_PROPOSAL_WIDTH * sizeof(HI_S32));
    SAMPLE_CHK_GOTO(HI_NULL == ps32AllBoxes, FAIL_0, "u32RoiCnt equals 0\n");

    pstStack = (NNIE_STACK_S*)malloc(sizeof(NNIE_STACK_S) * u32RoiCnt);
    SAMPLE_CHK_GOTO(HI_NULL == pstStack, FAIL_0, "u32RoiCnt equals 0\n");

    for (j = 0; j < u32ClassNum; j++)
    {
        for (i = 0; i < u32RoiCnt; i++)
        {
            fProposalXMin = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM] / 4096.0);
            fProposalYMin = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM + 1] / 4096.0);
            fProposalXMax = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM + 2] / 4096.0);
            fProposalYMax = (HI_FLOAT)(ps32Proposal[i * SVP_WK_COORDI_NUM + 3] / 4096.0);

            if (SAMPLE_RUNTIME_MODEL_TYPE_RFCN == enType)
            {
                fBBoxCenterXDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM] / 4096.0);
                fBBoxCenterYDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM + 1] / 4096.0);
                fBBoxWidthDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM + 2] / 4096.0);
                fBBoxHeightDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + SVP_WK_COORDI_NUM + 3] / 4096.0);
            }
            else
            {
                fBBoxCenterXDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM] / 4096.0);
                fBBoxCenterYDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM + 1] / 4096.0);
                fBBoxWidthDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM + 2] / 4096.0);
                fBBoxHeightDelta = (HI_FLOAT)(ps32BBox[i * u32BBoxStride / sizeof(HI_S32) + j * SVP_WK_COORDI_NUM + 3] / 4096.0);
            }

            fProposalWidth = fProposalXMax - fProposalXMin + 1;
            fProposalHeight = fProposalYMax - fProposalYMin + 1;
            fProposalCenterX = (HI_FLOAT)(fProposalXMin + fProposalWidth * 0.5);
            fProposalCenterY = (HI_FLOAT)(fProposalYMin + fProposalHeight * 0.5);
            fPredWidth = (HI_FLOAT)(fProposalWidth * exp(fBBoxWidthDelta));
            fPredHeight = (HI_FLOAT)(fProposalHeight * exp(fBBoxHeightDelta));
            fPredCenterX = fProposalCenterX + fProposalWidth * fBBoxCenterXDelta;
            fPredCenterY = fProposalCenterY + fProposalHeight * fBBoxCenterYDelta;
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH] = (HI_S32)(fPredCenterX - 0.5 * fPredWidth);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 1] = (HI_S32)(fPredCenterY - 0.5 * fPredHeight);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 2] = (HI_S32)(fPredCenterX + 0.5 * fPredWidth);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 3] = (HI_S32)(fPredCenterY + 0.5 * fPredHeight);
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 4] = ps32Score[i * u32ScoreStride / sizeof(HI_S32) + j];
            ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 5] = RPN_SUPPRESS_FALSE; // RPN Suppress
        }

        BboxClip_N(ps32AllBoxes, pstDataBlob->unShape.stWhc.u32Width, pstDataBlob->unShape.stWhc.u32Height, u32RoiCnt);
        ps32AllBoxesTmp = ps32AllBoxes;
        NonRecursiveArgQuickSort(ps32AllBoxesTmp, 0, u32RoiCnt - 1, pstStack, u32RoiCnt);
        NonMaxSuppression(ps32AllBoxesTmp, u32RoiCnt, (HI_U32)(0.7 * SVP_WK_QUANT_BASE));
        u32ResultBoxNum = 0;

        for (i = 0; i < u32RoiCnt; i++)
        {
            if (RPN_SUPPRESS_FALSE == ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 5]
                && ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 4] > 0.3 * SVP_WK_QUANT_BASE)
            {
#if SAMPLE_DEBUG
                printf("%d %f %d %d %d %d\n", j, ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 4] / 4096.0,
                       ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 0], ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 1],
                       ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 2], ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH + 3]);
#endif
                u32ResultBoxNum++;

                if (0 != j) // not background
                {
                    memcpy(&ps32ResultROI[*pu32ResultROICnt * SVP_WK_PROPOSAL_WIDTH],
                           &ps32AllBoxes[i * SVP_WK_PROPOSAL_WIDTH], sizeof(HI_S32)*SVP_WK_PROPOSAL_WIDTH);
                    *pu32ResultROICnt = *pu32ResultROICnt + 1;
                }
            }
        }

#if SAMPLE_DEBUG
        printf("class %d has %d boxes\n", j, u32ResultBoxNum);
#endif
    }

    SAMPLE_FREE(pstStack);
    SAMPLE_FREE(ps32AllBoxes);
    return HI_SUCCESS;
FAIL_0:
    SAMPLE_FREE(pstStack);
    SAMPLE_FREE(ps32AllBoxes);
    return HI_FAILURE;
}

#if 1

/*CNN Software parameter*/
typedef struct hiSAMPLE_RUNTIME_CNN_SOFTWARE_PARAM_S
{
    HI_U32 u32TopN;
    HI_RUNTIME_BLOB_S stGetTopN;
    HI_RUNTIME_MEM_S stAssistBuf;
} SAMPLE_RUNTIME_CNN_SOFTWARE_PARAM_S;

/*CNN GetTopN unit*/
typedef struct hiSAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S
{
    HI_U32   u32ClassId;
    HI_U32   u32Confidence;
} SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S;

static SAMPLE_RUNTIME_CNN_SOFTWARE_PARAM_S s_stCnnSoftwareParam = { 0 };

static HI_S32 SAMPLE_RUNTIME_GetTopN(HI_S32* ps32Fc, HI_U32 u32FcStride,
                                     HI_U32 u32ClassNum, HI_U32 u32BatchNum, HI_U32 u32TopN, HI_S32* ps32TmpBuf,
                                     HI_U32 u32TopNStride, HI_S32* ps32GetTopN)
{
    HI_U32 i = 0, j = 0, n = 0;
    HI_U32 u32Id = 0;
    HI_S32* ps32Score = NULL;
    SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S stTmp = { 0 };
    SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S* pstTopN = NULL;
    SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S* pstTmpBuf = (SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S*)ps32TmpBuf;

    for (n = 0; n < u32BatchNum; n++)
    {
        ps32Score = (HI_S32*)((HI_U8*)ps32Fc + n * u32FcStride);
        pstTopN = (SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S*)((HI_U8*)ps32GetTopN + n * u32TopNStride);

        for (i = 0; i < u32ClassNum; i++)
        {
            pstTmpBuf[i].u32ClassId = i;
            pstTmpBuf[i].u32Confidence = (HI_U32)ps32Score[i];
        }

        for (i = 0; i < u32TopN; i++)
        {
            u32Id = i;
            pstTopN[i].u32ClassId = pstTmpBuf[i].u32ClassId;
            pstTopN[i].u32Confidence = pstTmpBuf[i].u32Confidence;

            for (j = i + 1; j < u32ClassNum; j++)
            {
                if (pstTmpBuf[u32Id].u32Confidence < pstTmpBuf[j].u32Confidence)
                {
                    u32Id = j;
                }
            }

            stTmp.u32ClassId = pstTmpBuf[u32Id].u32ClassId;
            stTmp.u32Confidence = pstTmpBuf[u32Id].u32Confidence;

            if (i != u32Id)
            {
                pstTmpBuf[u32Id].u32ClassId = pstTmpBuf[i].u32ClassId;
                pstTmpBuf[u32Id].u32Confidence = pstTmpBuf[i].u32Confidence;
                pstTmpBuf[i].u32ClassId = stTmp.u32ClassId;
                pstTmpBuf[i].u32Confidence = stTmp.u32Confidence;

                pstTopN[i].u32ClassId = stTmp.u32ClassId;
                pstTopN[i].u32Confidence = stTmp.u32Confidence;
            }
        }
    }

    return HI_SUCCESS;
}

HI_S32 SAMPLE_RUNTIME_Cnn_PrintResult(HI_RUNTIME_BLOB_S* pstGetTopN, HI_U32 u32TopN)
{
    HI_U32 i = 0, j = 0;
    HI_U32* pu32Tmp = NULL;
    HI_U32 u32Stride = 0;
    float fProb = 0;
    SAMPLE_CHK_GOTO(NULL == pstGetTopN, FAIL_0, "Input error!\n");

    u32Stride = pstGetTopN->u32Stride;

    for (j = 0; j < pstGetTopN->u32Num; j++)
    {
        SAMPLE_LOG_PRINT("==== The %dth image top %u output info====\n", j, u32TopN);
        pu32Tmp = (HI_U32*)((HI_UL)(pstGetTopN->u64VirAddr + j * u32Stride));

        for (i = 0; i < u32TopN * 2; i += 2)
        {
            fProb = (float)pu32Tmp[i + 1] / 4096;
            SAMPLE_LOG_PRINT("Index:%d, Probability: %f(%d)\n",
                                  pu32Tmp[i], fProb, pu32Tmp[i + 1]);
        }
        SAMPLE_LOG_PRINT("==== The %dth image info end===\n", j);
    }

    return HI_SUCCESS;

FAIL_0:
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_Cnn_SoftwareParaInit(HI_RUNTIME_BLOB_S* pstDstBlob, SAMPLE_RUNTIME_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_U32 u32GetTopNMemSize = 0;
    HI_U32 u32GetTopNAssistBufSize = 0;
    HI_U32 u32GetTopNPerFrameSize = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32ClassNum = pstDstBlob->unShape.stWhc.u32Width;
    HI_U8* pu8VirAddr = NULL;

    /*get mem size*/
    u32GetTopNPerFrameSize = pstCnnSoftWarePara->u32TopN * sizeof(SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S);
    u32GetTopNMemSize = ALIGN16(u32GetTopNPerFrameSize) * pstDstBlob->u32Num;
    u32GetTopNAssistBufSize = u32ClassNum * sizeof(SAMPLE_RUNTIME_CNN_GETTOPN_UNIT_S);
    u32TotalSize = u32GetTopNMemSize + u32GetTopNAssistBufSize;


    SAMPLE_CHK_GOTO(0 == u32TotalSize, FAIL_0, "u32TotalSize equals 0\n");
    pu8VirAddr = (HI_U8*)malloc(u32TotalSize);
    SAMPLE_CHK_GOTO(NULL == pu8VirAddr, FAIL_0, "Malloc memory failed, number = %u!\n", pstDstBlob->u32Num);
    memset(pu8VirAddr, 0, u32TotalSize);

    /*init GetTopn */
    pstCnnSoftWarePara->stGetTopN.u32Num = pstDstBlob->u32Num;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Chn = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Height = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Width = u32GetTopNPerFrameSize / sizeof(HI_U32);
    pstCnnSoftWarePara->stGetTopN.u32Stride = ALIGN16(u32GetTopNPerFrameSize);
    pstCnnSoftWarePara->stGetTopN.u64PhyAddr = 0;
    pstCnnSoftWarePara->stGetTopN.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr);

    /*init AssistBuf */
    pstCnnSoftWarePara->stAssistBuf.u32Size = u32GetTopNAssistBufSize;
    pstCnnSoftWarePara->stAssistBuf.u64PhyAddr = 0;
    pstCnnSoftWarePara->stAssistBuf.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr) + u32GetTopNMemSize;

    return HI_SUCCESS;

FAIL_0:
    return HI_FAILURE;
}

void SAMPLE_RUNTIME_Cnn_SoftwareParaDeInit(SAMPLE_RUNTIME_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_U8* pu8VirAddr = (HI_U8*)((HI_UL)pstCnnSoftWarePara->stGetTopN.u64VirAddr);

    if (NULL != pu8VirAddr)
    {
        free(pu8VirAddr);
    }

    pstCnnSoftWarePara->stGetTopN.u64VirAddr = 0;
    pstCnnSoftWarePara->stAssistBuf.u64VirAddr = 0;

    return;
}

HI_S32 SAMPLE_RUNTIME_Cnn_GetTopN(HI_RUNTIME_BLOB_S* pstBlob,
                                  SAMPLE_RUNTIME_CNN_SOFTWARE_PARAM_S* pstCnnSoftwareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;

    s32Ret = SAMPLE_RUNTIME_GetTopN((HI_S32*)((HI_UL)pstBlob->u64VirAddr),
                                    pstBlob->u32Stride,
                                    pstBlob->unShape.stWhc.u32Width,
                                    pstBlob->u32Num,
                                    pstCnnSoftwareParam->u32TopN,
                                    (HI_S32*)((HI_UL)pstCnnSoftwareParam->stAssistBuf.u64VirAddr),
                                    pstCnnSoftwareParam->stGetTopN.u32Stride,
                                    (HI_S32*)((HI_UL)pstCnnSoftwareParam->stGetTopN.u64VirAddr));
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_GetTopN failed!\n");

FAIL_0:
    return s32Ret;
}

HI_S32 SAMPLE_RUNTIME_Cnn_TopN_Output(HI_RUNTIME_BLOB_S* pstDst, HI_U32 u32TopN)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_RUNTIME_BLOB_S* pstDstBlob = NULL;

    pstDstBlob = (HI_RUNTIME_BLOB_S*) & (pstDst[0]);
    s_stCnnSoftwareParam.u32TopN = u32TopN;

    s32Ret = SAMPLE_RUNTIME_Cnn_SoftwareParaInit(pstDstBlob, &s_stCnnSoftwareParam);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_Cnn_SoftwareParaInit failed!\n");

    s32Ret = SAMPLE_RUNTIME_Cnn_GetTopN(pstDstBlob, &s_stCnnSoftwareParam);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_Cnn_GetTopN failed!\n");

    s32Ret = SAMPLE_RUNTIME_Cnn_PrintResult(&(s_stCnnSoftwareParam.stGetTopN),
                                            s_stCnnSoftwareParam.u32TopN);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_RUNTIME_Cnn_PrintResult failed!\n");

FAIL_0:
    SAMPLE_RUNTIME_Cnn_SoftwareParaDeInit(&s_stCnnSoftwareParam);
    return s32Ret;
}
#endif

HI_S32 SAMPLE_RUNTIME_HiMemAlloc(HI_RUNTIME_MEM_S* pstMem, HI_BOOL bCached)
{
    HI_S32 s32Ret = HI_SUCCESS;

    pstMem->u64VirAddr = 0;
    s32Ret = SAMPLE_AllocMem(pstMem, bCached);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_Utils_AllocMem failed!\n");

    if (HI_FALSE == bCached)
    {
        return s32Ret;
    }

    s32Ret = SAMPLE_FlushCache(pstMem);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_Utils_FlushCache failed!\n");

    return s32Ret;
FAIL_0:
    SAMPLE_FreeMem(pstMem);
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_HiBlobAlloc(HI_RUNTIME_BLOB_S* pstBlob, HI_U32 u32BlobSize, HI_BOOL bCached)
{
    HI_S32 s32Ret = HI_SUCCESS;

    HI_RUNTIME_MEM_S stMem;
    memset(&stMem, 0, sizeof(stMem));
    stMem.u32Size = u32BlobSize;
    s32Ret = SAMPLE_AllocMem(&stMem, bCached);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_Utils_AllocMem failed!\n");
    pstBlob->u64PhyAddr = stMem.u64PhyAddr;
    pstBlob->u64VirAddr = stMem.u64VirAddr;

    if (HI_FALSE == bCached)
    {
        return s32Ret;
    }

    memset((HI_VOID*)((HI_UL)pstBlob->u64VirAddr), 0, u32BlobSize);

    s32Ret = SAMPLE_FlushCache(&stMem);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_Utils_FlushCache failed!\n");

    return s32Ret;
FAIL_0:
    (HI_VOID)SAMPLE_FreeMem(&stMem);
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_LoadModelFile(const HI_CHAR* pcModelFile, HI_RUNTIME_MEM_S* pstMemInfo)
{
    FILE* fp = HI_NULL;
    HI_S32  s32RuntimeWkLen = 0;
    HI_S32 s32Ret = HI_FAILURE;
    HI_CHAR acCanonicalPath[PATH_MAX+1] = {0};

#ifdef _WIN32
    SAMPLE_CHK_RETURN(strlen(pcModelFile) > PATH_MAX || HI_NULL == _fullpath(acCanonicalPath, pcModelFile, PATH_MAX), HI_FAILURE, "fullpath fail %s", pcModelFile);
#else
    SAMPLE_CHK_RETURN(strlen(pcModelFile) > PATH_MAX || HI_NULL == realpath(pcModelFile, acCanonicalPath), HI_FAILURE, "realpath fail %s", pcModelFile);
#endif
    fp = fopen(acCanonicalPath, "rb");
    SAMPLE_CHK_RETURN(NULL == fp, HI_FAILURE, "Open model file  %s failed!\n", pcModelFile);

    s32Ret = fseek(fp, 0L, SEEK_END);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, CLOSE_FILE, "SAMPLE_RUNTIME_MemAlloc failed!\n");

    s32RuntimeWkLen = ftell(fp);
    SAMPLE_CHK_GOTO(0 != s32RuntimeWkLen % 16, CLOSE_FILE, "Runtime WK Len %% 16 != 0 \n");

    SAMPLE_LOG_PRINT("Runtime WK Len: %d\n", s32RuntimeWkLen);

    SAMPLE_CHK_GOTO(0 != fseek(fp, 0L, SEEK_SET), CLOSE_FILE, "fseek fail");

    pstMemInfo->u32Size = s32RuntimeWkLen;
    s32Ret = SAMPLE_RUNTIME_HiMemAlloc(pstMemInfo, HI_FALSE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, CLOSE_FILE, "SAMPLE_RUNTIME_MemAlloc failed!\n");

    s32Ret = (HI_S32)fread((HI_VOID*)((HI_UL)pstMemInfo->u64VirAddr), s32RuntimeWkLen, 1, fp);
    SAMPLE_CHK_GOTO(1 != s32Ret, FREE_MEM, "Read runtime WK failed!\n");

    fclose(fp);
    return HI_SUCCESS;
FREE_MEM:
    SAMPLE_FreeMem(pstMemInfo);
CLOSE_FILE:
    fclose(fp);
    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_SetBlob(HI_RUNTIME_BLOB_S* pstBlob,
        HI_RUNTIME_BLOB_TYPE_E enType,
        HI_U32 u32Num,
        HI_U32 u32Width,
        HI_U32 u32Height,
        HI_U32 u32Chn,
        HI_U32 u32Align)
{
    HI_S32 s32Ret = HI_FAILURE;
    HI_U32 u32BlobSize = 0;

    pstBlob->enBlobType = enType;
    pstBlob->u32Num = u32Num;
    if (enType == HI_RUNTIME_BLOB_TYPE_VEC_S32)
    {
        pstBlob->unShape.stWhc.u32Width = u32Chn;
        pstBlob->unShape.stWhc.u32Chn = u32Width;
    }
    else
    {
        pstBlob->unShape.stWhc.u32Width = u32Width;
        pstBlob->unShape.stWhc.u32Chn = u32Chn;
    }
    pstBlob->unShape.stWhc.u32Height = u32Height;
    SAMPLE_DATA_GetStride(enType, pstBlob->unShape.stWhc.u32Width, u32Align, &(pstBlob->u32Stride));

    SAMPLE_CHK_GOTO(((HI_U64)pstBlob->u32Num * pstBlob->u32Stride * pstBlob->unShape.stWhc.u32Height * pstBlob->unShape.stWhc.u32Chn > (HI_U32)-1), FAIL_0, "the blobsize is too large [%llu]\n", (HI_U64)pstBlob->u32Num * pstBlob->u32Stride * pstBlob->unShape.stWhc.u32Height * pstBlob->unShape.stWhc.u32Chn);
    u32BlobSize = SAMPLE_DATA_GetBlobSize(pstBlob->u32Stride, u32Num, u32Height, pstBlob->unShape.stWhc.u32Chn);

    s32Ret = SAMPLE_RUNTIME_HiBlobAlloc(pstBlob, u32BlobSize, HI_TRUE);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_Utils_AllocMem failed!\n");
FAIL_0:
    return s32Ret;
}

HI_S32 SAMPLE_RUNTIME_ReadSrcFile(const HI_CHAR* pcSrcFile, HI_RUNTIME_BLOB_S* pstSrcBlob)
{
    HI_U32 c, h;
    HI_U8* pu8Ptr = NULL;
    FILE* imgFp = NULL;
    HI_U32 s32Ret = HI_FAILURE;
    HI_U32 u32BlobSize = 0;
    HI_RUNTIME_MEM_S stMem;
    HI_CHAR acCanonicalPath[PATH_MAX+1] = {0};

#ifdef _WIN32
    SAMPLE_CHK_RETURN(strlen(pcSrcFile) > PATH_MAX || HI_NULL == _fullpath(acCanonicalPath, pcSrcFile, PATH_MAX), HI_FAILURE, "fullpath fail %s", pcSrcFile);
#else
    SAMPLE_CHK_RETURN(strlen(pcSrcFile) > PATH_MAX || HI_NULL == realpath(pcSrcFile, acCanonicalPath), HI_FAILURE, "realpath fail %s", pcSrcFile);
#endif
    pu8Ptr = (HI_U8*)((HI_UL)(pstSrcBlob->u64VirAddr));
    imgFp = fopen(acCanonicalPath, "rb");
    SAMPLE_CHK_GOTO(NULL == imgFp, FAIL_0, "open img fp error[%s]\n", pcSrcFile);

    for (c = 0; c < pstSrcBlob->unShape.stWhc.u32Chn; c++)
    {
        for (h = 0; h < pstSrcBlob->unShape.stWhc.u32Height; h++)
        {
            s32Ret = (HI_S32)fread(pu8Ptr, pstSrcBlob->unShape.stWhc.u32Width * sizeof(HI_U8), 1, imgFp);
            SAMPLE_CHK_GOTO(1 != s32Ret, FAIL_0, "fread failed, (c,h)=(%d,%d)!", c, h);

            pu8Ptr += pstSrcBlob->u32Stride;
        }
    }

    fclose(imgFp);
    imgFp = HI_NULL;

    u32BlobSize = SAMPLE_DATA_GetBlobSize(pstSrcBlob->u32Stride, pstSrcBlob->u32Num, pstSrcBlob->unShape.stWhc.u32Height, pstSrcBlob->unShape.stWhc.u32Chn);
    stMem.u64PhyAddr = pstSrcBlob->u64PhyAddr;
    stMem.u64VirAddr = pstSrcBlob->u64VirAddr;
    stMem.u32Size = u32BlobSize;
    s32Ret = SAMPLE_FlushCache(&stMem);
    SAMPLE_CHK_GOTO(HI_SUCCESS != s32Ret, FAIL_0, "SAMPLE_Utils_FlushCache failed!\n");

    return HI_SUCCESS;
FAIL_0:

    if (HI_NULL != imgFp)
    { fclose(imgFp); }

    return HI_FAILURE;
}

HI_S32 SAMPLE_RUNTIME_ReadConfig(const HI_CHAR* pcConfigFile, HI_CHAR acBuff[], HI_U32 u32BufSize)
{
    HI_CHAR acCanonicalPath[PATH_MAX+1] = {0};
    FILE *f = NULL;
    HI_U32 u32ReadSize  = 0;
    memset(acBuff, 0, u32BufSize);
#ifdef _WIN32
    SAMPLE_CHK_RETURN(strlen(pcConfigFile) > PATH_MAX || HI_NULL == _fullpath(acCanonicalPath, pcConfigFile, PATH_MAX), HI_FAILURE, "fullpath fail %s", pcConfigFile);
#else
    SAMPLE_CHK_RETURN(strlen(pcConfigFile) > PATH_MAX || HI_NULL == realpath(pcConfigFile, acCanonicalPath), HI_FAILURE, "realpath fail %s", pcConfigFile);
#endif
    f = fopen(acCanonicalPath, "r");
    SAMPLE_CHK_RETURN(NULL == f, HI_FAILURE, "config file %s not exists\n", acCanonicalPath);
    u32ReadSize = fread(acBuff, 1, u32BufSize-1, f);
    if ((u32ReadSize != (u32BufSize - 1)) && !feof(f))
    {
        fclose(f);
        return HI_FAILURE;
    }
    fclose(f);
    return HI_SUCCESS;
}

HI_DOUBLE compute_output_w(HI_DOUBLE x1, HI_DOUBLE x2)
{
    const HI_DOUBLE bbox_width = (x2 - x1);
    const HI_DOUBLE output_width = 2 * bbox_width;

    return max(1.0, output_width);
}

HI_DOUBLE compute_output_h(HI_DOUBLE y1, HI_DOUBLE y2)
{
    const HI_DOUBLE bbox_height = (y2 - y1);
    const HI_DOUBLE output_height = 2 * bbox_height;

    return max(1.0, output_height);
}

HI_VOID computeCropLocation(const BondingBox_s *pstTightBbox, HI_DOUBLE dWidth, HI_DOUBLE dHeight, BondingBox_s* pstLocationBbox)
{
    const HI_DOUBLE centerx = (pstTightBbox->x1 + pstTightBbox->x2) / 2;
    const HI_DOUBLE centery = (pstTightBbox->y1 + pstTightBbox->y2) / 2;

    const HI_DOUBLE width = dWidth;
    const HI_DOUBLE height = dHeight;

    const HI_DOUBLE output_width = compute_output_w(pstTightBbox->x1, pstTightBbox->x2);
    const HI_DOUBLE output_height = compute_output_h(pstTightBbox->y1, pstTightBbox->y2);

    const HI_DOUBLE roi_left = max(0.0, centerx - output_width / 2);
    const HI_DOUBLE roi_bottom = max(0.0, centery - output_height / 2);


    const HI_DOUBLE left_half = min(output_width / 2, centerx);

    const HI_DOUBLE right_half = min(output_width / 2, width - centerx);

    const HI_DOUBLE roi_width = max(1.0, left_half + right_half);

    const HI_DOUBLE top_half = min(output_height / 2, centery);

    const HI_DOUBLE bottom_half = min(output_height / 2, height - centery);

    const HI_DOUBLE roi_height = max(1.0, top_half + bottom_half);

    pstLocationBbox->x1 = roi_left;
    pstLocationBbox->y1 = roi_bottom;
    pstLocationBbox->x2 = roi_left + roi_width;
    pstLocationBbox->y2 = roi_bottom + roi_height;

}

HI_DOUBLE compute_edge_x(HI_DOUBLE x1, HI_DOUBLE x2)
{
    const HI_DOUBLE output_width = compute_output_w(x1, x2);
    const HI_DOUBLE centerx = (x1 + x2) / 2;

    return max(0.0, output_width / 2 - centerx);
}

HI_DOUBLE compute_edge_y(HI_DOUBLE y1, HI_DOUBLE y2)
{
    const HI_DOUBLE output_height = compute_output_h(y1, y2);
    const HI_DOUBLE centery = (y1 + y2) / 2;

    return max(0.0, output_height / 2 - centery);
}

