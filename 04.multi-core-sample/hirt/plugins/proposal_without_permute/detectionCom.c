#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "back/hi_type.h"
#include "detectionCom.h"
#include "stdio.h"

HI_FLOAT af32ExpCoef[10][16] = {
    { (HI_FLOAT)1, (HI_FLOAT)1.00024, (HI_FLOAT)1.00049, (HI_FLOAT)1.00073, (HI_FLOAT)1.00098, (HI_FLOAT)1.00122, (HI_FLOAT)1.00147, (HI_FLOAT)1.00171, (HI_FLOAT)1.00196, (HI_FLOAT)1.0022, (HI_FLOAT)1.00244, (HI_FLOAT)1.00269, (HI_FLOAT)1.00293, (HI_FLOAT)1.00318, (HI_FLOAT)1.00342, (HI_FLOAT)1.00367 },
    { (HI_FLOAT)1, (HI_FLOAT)1.00391, (HI_FLOAT)1.00784, (HI_FLOAT)1.01179, (HI_FLOAT)1.01575, (HI_FLOAT)1.01972, (HI_FLOAT)1.02371, (HI_FLOAT)1.02772, (HI_FLOAT)1.03174, (HI_FLOAT)1.03578, (HI_FLOAT)1.03984, (HI_FLOAT)1.04391, (HI_FLOAT)1.04799, (HI_FLOAT)1.05209, (HI_FLOAT)1.05621, (HI_FLOAT)1.06034 },
    { (HI_FLOAT)1, (HI_FLOAT)1.06449, (HI_FLOAT)1.13315, (HI_FLOAT)1.20623, (HI_FLOAT)1.28403, (HI_FLOAT)1.36684, (HI_FLOAT)1.45499, (HI_FLOAT)1.54883, (HI_FLOAT)1.64872, (HI_FLOAT)1.75505, (HI_FLOAT)1.86825, (HI_FLOAT)1.98874, (HI_FLOAT)2.117, (HI_FLOAT)2.25353, (HI_FLOAT)2.39888, (HI_FLOAT)2.55359 },
    { (HI_FLOAT)1, (HI_FLOAT)2.71828, (HI_FLOAT)7.38906, (HI_FLOAT)20.0855, (HI_FLOAT)54.5981, (HI_FLOAT)148.413, (HI_FLOAT)403.429, (HI_FLOAT)1096.63, (HI_FLOAT)2980.96, (HI_FLOAT)8103.08, (HI_FLOAT)22026.5, (HI_FLOAT)59874.1, (HI_FLOAT)162755, (HI_FLOAT)442413, (HI_FLOAT)1.2026e+006, (HI_FLOAT)3.26902e+006 },
    { (HI_FLOAT)1, (HI_FLOAT)8.88611e+006, (HI_FLOAT)7.8963e+013, (HI_FLOAT)7.01674e+020, (HI_FLOAT)6.23515e+027, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034, (HI_FLOAT)5.54062e+034 },
    { (HI_FLOAT)1, (HI_FLOAT)0.999756, (HI_FLOAT)0.999512, (HI_FLOAT)0.999268, (HI_FLOAT)0.999024, (HI_FLOAT)0.99878, (HI_FLOAT)0.998536, (HI_FLOAT)0.998292, (HI_FLOAT)0.998049, (HI_FLOAT)0.997805, (HI_FLOAT)0.997562, (HI_FLOAT)0.997318, (HI_FLOAT)0.997075, (HI_FLOAT)0.996831, (HI_FLOAT)0.996588, (HI_FLOAT)0.996345 },
    { (HI_FLOAT)1, (HI_FLOAT)0.996101, (HI_FLOAT)0.992218, (HI_FLOAT)0.98835, (HI_FLOAT)0.984496, (HI_FLOAT)0.980658, (HI_FLOAT)0.976835, (HI_FLOAT)0.973027, (HI_FLOAT)0.969233, (HI_FLOAT)0.965455, (HI_FLOAT)0.961691, (HI_FLOAT)0.957941, (HI_FLOAT)0.954207, (HI_FLOAT)0.950487, (HI_FLOAT)0.946781, (HI_FLOAT)0.94309 },
    { (HI_FLOAT)1, (HI_FLOAT)0.939413, (HI_FLOAT)0.882497, (HI_FLOAT)0.829029, (HI_FLOAT)0.778801, (HI_FLOAT)0.731616, (HI_FLOAT)0.687289, (HI_FLOAT)0.645649, (HI_FLOAT)0.606531, (HI_FLOAT)0.569783, (HI_FLOAT)0.535261, (HI_FLOAT)0.502832, (HI_FLOAT)0.472367, (HI_FLOAT)0.443747, (HI_FLOAT)0.416862, (HI_FLOAT)0.391606 },
    { (HI_FLOAT)1, (HI_FLOAT)0.367879, (HI_FLOAT)0.135335, (HI_FLOAT)0.0497871, (HI_FLOAT)0.0183156, (HI_FLOAT)0.00673795, (HI_FLOAT)0.00247875, (HI_FLOAT)0.000911882, (HI_FLOAT)0.000335463, (HI_FLOAT)0.00012341, (HI_FLOAT)4.53999e-005, (HI_FLOAT)1.67017e-005, (HI_FLOAT)6.14421e-006, (HI_FLOAT)2.26033e-006, (HI_FLOAT)8.31529e-007, (HI_FLOAT)3.05902e-007 },
    { (HI_FLOAT)1, (HI_FLOAT)1.12535e-007, (HI_FLOAT)1.26642e-014, (HI_FLOAT)1.42516e-021, (HI_FLOAT)1.60381e-028, (HI_FLOAT)1.80485e-035, (HI_FLOAT)2.03048e-042, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0, (HI_FLOAT)0 }
};

/*********************************************************
Function: QuickExp
Description: Do QuickExp...
*********************************************************/
HI_FLOAT QuickExp(HI_U32 u32X)
{
    if (u32X & 0x80000000)
    {
        u32X = ~u32X + 0x00000001;
        return af32ExpCoef[5][u32X & 0x0000000F] * af32ExpCoef[6][(u32X >> 4) & 0x0000000F] * af32ExpCoef[7][(u32X >> 8) & 0x0000000F] * af32ExpCoef[8][(u32X >> 12) & 0x0000000F] * af32ExpCoef[9][(u32X >> 16) & 0x0000000F];
    }
    else
    {
        return af32ExpCoef[0][u32X & 0x0000000F] * af32ExpCoef[1][(u32X >> 4) & 0x0000000F] * af32ExpCoef[2][(u32X >> 8) & 0x0000000F] * af32ExpCoef[3][(u32X >> 12) & 0x0000000F] * af32ExpCoef[4][(u32X >> 16) & 0x0000000F];
    }
}

/*********************************************************
Function: SoftMax
Description: Do softmax on a vector of length s32ArraySize
*********************************************************/
HI_S32 SoftMax(HI_S32 *ps32BgSrc, HI_S32 *ps32FgSrc, HI_S32 s32ArraySize)
{
    /***** define parameters ****/
    HI_S32 s32BgSrc = *ps32BgSrc;
    HI_S32 s32FgSrc = *ps32FgSrc;
    HI_S32 s32Max = s32BgSrc > s32FgSrc ? s32BgSrc : s32FgSrc;

    HI_FLOAT f32BgExp = QuickExp(s32BgSrc-s32Max);
    HI_FLOAT f32FgExp = QuickExp(s32FgSrc-s32Max);
    HI_FLOAT f32Sum = f32BgExp+f32FgExp;

    *ps32BgSrc = (f32BgExp/f32Sum)*SVP_WK_QUANT_BASE;
    *ps32FgSrc = (f32FgExp/f32Sum)*SVP_WK_QUANT_BASE;

    return HI_SUCCESS;
}

HI_S32 SoftMax_N(HI_S32 *as32BgSrc, HI_S32 *as32FgSrc, HI_S32 s32ArraySize, HI_U32 u32Num)
{
    HI_S32 s32Ret = HI_FAILURE;
    for (HI_U32 i = 0; i < u32Num; i++)
    {
        s32Ret = SoftMax(&as32BgSrc[i], &as32FgSrc[i], s32ArraySize);
        SVP_FALSE_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);
    }
    return HI_SUCCESS;
}

/*********************************************************
Function: Overlap
Description: Calculate the IOU of two bboxes
*********************************************************/
static HI_S32 Overlap(HI_S32 s32XMin1, HI_S32 s32YMin1, HI_S32 s32XMax1, HI_S32 s32YMax1, HI_S32 s32XMin2,
    HI_S32 s32YMin2, HI_S32 s32XMax2, HI_S32 s32YMax2, HI_S32* s32AreaSum, HI_S32* s32AreaInter,
    HI_S32 s32Area1, HI_S32 s32Area2)
{
    /*** Check the input, and change the Return value  ***/
    HI_S32 s32Inter = 0;
    HI_S32 s32Total = 0;
    HI_S32 s32XMin = 0;
    HI_S32 s32YMin = 0;
    HI_S32 s32XMax = 0;
    HI_S32 s32YMax = 0;
    HI_S32 s32InterWidth = 0;
    HI_S32 s32InterHeight = 0;

    s32XMin = SVP_MAX(s32XMin1, s32XMin2);
    s32YMin = SVP_MAX(s32YMin1, s32YMin2);
    s32XMax = SVP_MIN(s32XMax1, s32XMax2);
    s32YMax = SVP_MIN(s32YMax1, s32YMax2);

    s32InterWidth = s32XMax - s32XMin + 1;
    s32InterHeight = s32YMax - s32YMin + 1;

    s32InterWidth = (s32InterWidth >= 0) ? s32InterWidth : 0;
    s32InterHeight = (s32InterHeight >= 0) ? s32InterHeight : 0;

    s32Inter = s32InterWidth * s32InterHeight;

    s32Total = s32Area1 + s32Area2 - s32Inter;

    *s32AreaSum = s32Total;
    *s32AreaInter = s32Inter;

    return HI_SUCCESS;
}

/**************************************************
Function: Argswap
Description: used in NonRecursiveQuickSort
***************************************************/
HI_S32 Argswap(HI_S32* ps32Src1, HI_S32* ps32Src2)
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

/**************************************************
Function: NonRecursiveArgQuickSort
Description: sort with NonRecursiveArgQuickSort
***************************************************/
HI_S32 NonRecursiveArgQuickSort(HI_S32* aResultArray,
    HI_S32 s32Low, HI_S32 s32High, NNIE_STACK_S *pstStack, HI_U32 u32MaxNum)
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
                Argswap(&aResultArray[i*SVP_WK_PROPOSAL_WIDTH], &aResultArray[j*SVP_WK_PROPOSAL_WIDTH]);
                i++;
            }

            while ((i < j) && (s32KeyConfidence < aResultArray[i*SVP_WK_PROPOSAL_WIDTH + 4]))
            {
                i++;
            }
            if (i < j)
            {
                Argswap(&aResultArray[i*SVP_WK_PROPOSAL_WIDTH], &aResultArray[j*SVP_WK_PROPOSAL_WIDTH]);
                j--;
            }
        }

        if(s32Low<=u32MaxNum)
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
        }
    }
    return HI_SUCCESS;
}

/**************************************************
Function: NonMaxSuppression
Description: proposal NMS u32NmsThresh
***************************************************/
HI_S32 NonMaxSuppression(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh, HI_U32 u32MaxRoiNum)
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
    HI_BOOL bNoOverlap = HI_FALSE;
    HI_U32 u32Num = 0;

    SVP_FALSE_CHECK((0 != u32NumAnchors), HI_FAILURE);

    HI_S32 *ps32AreaEachAnchor = (HI_S32*)malloc(u32NumAnchors*sizeof(HI_S32));
    SVP_FALSE_CHECK(NULL != ps32AreaEachAnchor, HI_FAILURE);
    memset(ps32AreaEachAnchor, 0x0,u32NumAnchors*sizeof(HI_S32));
    for (HI_U32 i = 0; i < u32NumAnchors; i++)
    {
        if (RPN_SUPPRESS_FALSE == pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5])
        {
            s32XMin1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i];
            s32YMin1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 1];
            s32XMax1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 2];
            s32YMax1 = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 3];
            ps32AreaEachAnchor[i] = (s32YMax1-s32YMin1+1)*(s32XMax1-s32XMin1+1);
        }
    }
    for (HI_U32 i = 0; i < u32NumAnchors && u32Num < u32MaxRoiNum; i++)
    {
        if (RPN_SUPPRESS_FALSE == pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5])
        {
            u32Num++;
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

                    bNoOverlap = (s32XMin2>s32XMax1) || (s32XMax2<s32XMin1) || (s32YMin2>s32YMax1) || (s32YMax2<s32YMin1) ? HI_TRUE : HI_FALSE;
                    if (bNoOverlap)
                    {
                        continue;
                    }

                    Overlap(s32XMin1, s32YMin1, s32XMax1, s32YMax1, s32XMin2, s32YMin2, s32XMax2, s32YMax2,
                            &s32AreaTotal, &s32AreaInter, ps32AreaEachAnchor[i], ps32AreaEachAnchor[j]);
                    if (s32AreaInter * SVP_WK_QUANT_BASE >(HI_S32)u32NmsThresh * s32AreaTotal)
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
    free(ps32AreaEachAnchor);
    return HI_SUCCESS;
}

/**************************************************
Function: FilterLowScoreBbox
Description: remove low conf score proposal bbox
***************************************************/
HI_S32 FilterLowScoreBbox(HI_S32* pu32Proposals, HI_U32 u32NumAnchors, HI_U32 u32NmsThresh, HI_U32 u32FilterThresh, HI_U32* u32NumAfterFilter)
{
    HI_U32 u32ProposalCnt = 0;
    HI_U32 i = 0;

    u32ProposalCnt = u32NumAnchors;

    if (u32FilterThresh > 0)
    {
        for (i = 0; i < u32NumAnchors; i++)
        {
            if (pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 4] < (HI_S32)u32FilterThresh)
            {
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5] = RPN_SUPPRESS_TRUE;
            }
        }

        u32ProposalCnt = 0;
        for (i = 0; i < u32NumAnchors; i++)
        {
            if (RPN_SUPPRESS_FALSE == pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5])
            {
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * u32ProposalCnt + 0] = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 0];
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * u32ProposalCnt + 1] = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 1];
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * u32ProposalCnt + 2] = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 2];
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * u32ProposalCnt + 3] = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 3];
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * u32ProposalCnt + 4] = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 4];
                pu32Proposals[SVP_WK_PROPOSAL_WIDTH * u32ProposalCnt + 5] = pu32Proposals[SVP_WK_PROPOSAL_WIDTH * i + 5];
                u32ProposalCnt++;
            }
        }
    }

    *u32NumAfterFilter = u32ProposalCnt;
    return HI_SUCCESS;
}

/**************************************************
Function: generate Base Anchors
Description: generate Base Anchors by give miniSize, ratios, and scales
***************************************************/
HI_S32 GenBaseAnchor(
    HI_FLOAT* pf32RatioAnchors, const HI_U32* pu32Ratios, HI_U32 u32NumRatioAnchors,
    HI_FLOAT* pf32ScaleAnchors, const HI_U32* pu32Scales, HI_U32 u32NumScaleAnchors,
    const HI_U32* au32BaseAnchor)
{
    /********************* Generate the base anchor ***********************/
    HI_FLOAT f32BaseW = (HI_FLOAT)(au32BaseAnchor[2] - au32BaseAnchor[0] + 1);
    HI_FLOAT f32BaseH = (HI_FLOAT)(au32BaseAnchor[3] - au32BaseAnchor[1] + 1);
    HI_FLOAT f32BaseXCtr = (HI_FLOAT)(au32BaseAnchor[0] + ((f32BaseW - 1) * 0.5));
    HI_FLOAT f32BaseYCtr = (HI_FLOAT)(au32BaseAnchor[1] + ((f32BaseH - 1) * 0.5));

    /*************** Generate Ratio Anchors for the base anchor ***********/
    HI_FLOAT f32Ratios = 0.0f;
    HI_FLOAT f32SizeRatios = 0.0f;
    HI_FLOAT f32Size = f32BaseW * f32BaseH;

    for (HI_U32 i = 0; i < u32NumRatioAnchors; i++)
    {
        f32Ratios = (HI_FLOAT)pu32Ratios[i] / SVP_WK_QUANT_BASE;
        f32SizeRatios = f32Size / f32Ratios;
        f32BaseW = (HI_FLOAT)SAFE_ROUND(sqrt(f32SizeRatios));
        f32BaseH = (HI_FLOAT)SAFE_ROUND(f32BaseW * f32Ratios);

        pf32RatioAnchors[i*SVP_WK_COORDI_NUM + 0] = f32BaseXCtr - (f32BaseW - 1) * 0.5f;
        pf32RatioAnchors[i*SVP_WK_COORDI_NUM + 1] = f32BaseYCtr - (f32BaseH - 1) * 0.5f;
        pf32RatioAnchors[i*SVP_WK_COORDI_NUM + 2] = f32BaseXCtr + (f32BaseW - 1) * 0.5f;
        pf32RatioAnchors[i*SVP_WK_COORDI_NUM + 3] = f32BaseYCtr + (f32BaseH - 1) * 0.5f;
    }

    /********* Generate Scale Anchors for each Ratio Anchor **********/
    /* Generate Scale Anchors for one pixel */
    HI_FLOAT f32Scales = 0.0f;
    for (HI_U32 i = 0; i < u32NumRatioAnchors; i++)
    {
        for (HI_U32 j = 0; j < u32NumScaleAnchors; j++)
        {
            f32BaseW = pf32RatioAnchors[2] - pf32RatioAnchors[0] + 1;
            f32BaseH = pf32RatioAnchors[3] - pf32RatioAnchors[1] + 1;
            f32BaseXCtr = pf32RatioAnchors[0] + (f32BaseW - 1) * 0.5f;
            f32BaseYCtr = pf32RatioAnchors[1] + (f32BaseH - 1) * 0.5f;

            f32Scales = (HI_FLOAT)pu32Scales[j] / SVP_WK_QUANT_BASE;
            pf32ScaleAnchors[0] = f32BaseXCtr - (f32BaseW * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors[1] = f32BaseYCtr - (f32BaseH * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors[2] = f32BaseXCtr + (f32BaseW * f32Scales - 1) * 0.5f;
            pf32ScaleAnchors[3] = f32BaseYCtr + (f32BaseH * f32Scales - 1) * 0.5f;

            pf32ScaleAnchors += SVP_WK_COORDI_NUM;
        }
        pf32RatioAnchors += SVP_WK_COORDI_NUM;
    }

    return HI_SUCCESS;
}

/**************************************************
Function: SetAnchorInPixel
Description: set base anchor to origin pic point based on pf32ScaleAnchors
***************************************************/
HI_S32 SetAnchorInPixel(
    HI_S32* ps32Anchors,
    const HI_FLOAT* pf32ScaleAnchors,
    HI_U32 u32ConvHeight,
    HI_U32 u32ConvWidth,
    HI_U32 u32NumAnchorPerPixel,
    HI_U32 u32SpatialScale)
{
    HI_U32 u32anchorCentorX = 0;
    HI_U32 u32anchorCentorY = 0;
    HI_U32 u32ScaleAnchorIndexBase = 0;

    /******************* Copy the anchors to every pixel in the feature map ******************/
    HI_FLOAT f32PixelInterval = SVP_WK_QUANT_BASE / (HI_FLOAT)u32SpatialScale;

    for (HI_U32 h = 0; h < u32ConvHeight; h++)
    {
        for (HI_U32 w = 0; w < u32ConvWidth; w++)
        {
            u32anchorCentorX = (HI_U32)(w * f32PixelInterval);
            u32anchorCentorY = (HI_U32)(h * f32PixelInterval);

            for (HI_U32 n = 0; n < u32NumAnchorPerPixel; n++)
            {
                u32ScaleAnchorIndexBase = n * SVP_WK_COORDI_NUM;
                ps32Anchors[0] = (HI_S32)(u32anchorCentorX + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 0]);
                ps32Anchors[1] = (HI_S32)(u32anchorCentorY + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 1]);
                ps32Anchors[2] = (HI_S32)(u32anchorCentorX + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 2]);
                ps32Anchors[3] = (HI_S32)(u32anchorCentorY + pf32ScaleAnchors[u32ScaleAnchorIndexBase + 3]);

                ps32Anchors += SVP_WK_COORDI_NUM;
            }
        }
    }

    return HI_SUCCESS;
}

/************************* BBox Transform *****************************/
/* use parameters from Conv3 to adjust the coordinates of anchors */
HI_S32 BboxTransform(
    HI_S32* ps32Proposals,
    HI_S32* ps32Anchors,
    HI_S32* ps32BboxDelta,
    HI_S32* ps32Scores)
{
    HI_S32 s32ProposalWidth = ps32Anchors[2] - ps32Anchors[0] + 1;
    HI_S32 s32ProposalHeight = ps32Anchors[3] - ps32Anchors[1] + 1;
    HI_S32 s32ProposalCenterX = (HI_S32)(ps32Anchors[0] + s32ProposalWidth * 0.5f);
    HI_S32 s32ProposalCenterY = (HI_S32)(ps32Anchors[1] + s32ProposalHeight * 0.5f);
    HI_S32 s32PredCenterX = (HI_S32)(((HI_FLOAT)ps32BboxDelta[0] / SVP_WK_QUANT_BASE) * s32ProposalWidth + s32ProposalCenterX);
    HI_S32 s32PredCenterY = (HI_S32)(((HI_FLOAT)ps32BboxDelta[1] / SVP_WK_QUANT_BASE) * s32ProposalHeight + s32ProposalCenterY);

    HI_S32 s32PredW = (HI_S32)(s32ProposalWidth  * QuickExp(ps32BboxDelta[2]));
    HI_S32 s32PredH = (HI_S32)(s32ProposalHeight * QuickExp(ps32BboxDelta[3]));

    ps32Proposals[0] = (HI_S32)(s32PredCenterX - 0.5f * s32PredW);
    ps32Proposals[1] = (HI_S32)(s32PredCenterY - 0.5f * s32PredH);
    ps32Proposals[2] = (HI_S32)(s32PredCenterX + 0.5f * s32PredW);
    ps32Proposals[3] = (HI_S32)(s32PredCenterY + 0.5f * s32PredH);
    ps32Proposals[4] = *ps32Scores;
    ps32Proposals[5] = RPN_SUPPRESS_FALSE;

    return HI_SUCCESS;
}

HI_S32 BboxTransform_N(
    HI_S32* ps32Proposals,
    HI_S32* ps32Anchors,
    HI_S32* ps32BboxDelta,
    HI_S32* ps32Scores,
    HI_U32 u32NumAnchors)
{
    HI_S32 s32Ret = HI_FAILURE;
    for (HI_U32 i = 0; i < u32NumAnchors; i++)
    {
        s32Ret = BboxTransform(
            &ps32Proposals[i*SVP_WK_PROPOSAL_WIDTH],
            &ps32Anchors[i*SVP_WK_COORDI_NUM],
            &ps32BboxDelta[i*SVP_WK_COORDI_NUM],
            &ps32Scores[i]);  /* be careful this +1 */
        SVP_FALSE_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);
    }
    return HI_SUCCESS;
}

/* all float type para */
HI_S32 BboxTransform_FLOAT(
    HI_FLOAT* pf32Proposals,
    HI_FLOAT* pf32Anchors,
    HI_FLOAT* pf32BboxDelta,
    HI_FLOAT* pf32Scores)
{
    HI_FLOAT s32ProposalWidth = pf32Anchors[2] - pf32Anchors[0] + 1;
    HI_FLOAT s32ProposalHeight = pf32Anchors[3] - pf32Anchors[1] + 1;
    HI_FLOAT s32ProposalCenterX = (pf32Anchors[0] + s32ProposalWidth * 0.5f);
    HI_FLOAT s32ProposalCenterY = (pf32Anchors[1] + s32ProposalHeight * 0.5f);

    HI_FLOAT s32PredCenterX = (pf32BboxDelta[0] / SVP_WK_QUANT_BASE) * s32ProposalWidth + s32ProposalCenterX;
    HI_FLOAT s32PredCenterY = (pf32BboxDelta[1] / SVP_WK_QUANT_BASE) * s32ProposalHeight + s32ProposalCenterY;
    HI_FLOAT s32PredW = s32ProposalWidth  * QuickExp((HI_S32)pf32BboxDelta[2]);
    HI_FLOAT s32PredH = s32ProposalHeight * QuickExp((HI_S32)pf32BboxDelta[3]);

    pf32Proposals[0] = s32PredCenterX - 0.5f * s32PredW;
    pf32Proposals[1] = (s32PredCenterY - 0.5f * s32PredH);
    pf32Proposals[2] = (s32PredCenterX + 0.5f * s32PredW);
    pf32Proposals[3] = (s32PredCenterY + 0.5f * s32PredH);
    pf32Proposals[4] = (*pf32Scores * SVP_WK_QUANT_BASE);
    pf32Proposals[5] = RPN_SUPPRESS_FALSE;

    return HI_SUCCESS;
}

/**************************************************
Function: BboxClip
Description: clip proposal bbox out of origin image range
***************************************************/
HI_S32 SizeClip(HI_S32 s32inputSize, HI_S32 s32sizeMin, HI_S32 s32sizeMax)
{
    return SVP_MAX(SVP_MIN(s32inputSize, s32sizeMax), s32sizeMin);
}

HI_S32 BboxClip(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH)
{
    ps32Proposals[0] = SizeClip(ps32Proposals[0], 0, (HI_S32)u32ImageW - 1);
    ps32Proposals[1] = SizeClip(ps32Proposals[1], 0, (HI_S32)u32ImageH - 1);
    ps32Proposals[2] = SizeClip(ps32Proposals[2], 0, (HI_S32)u32ImageW - 1);
    ps32Proposals[3] = SizeClip(ps32Proposals[3], 0, (HI_S32)u32ImageH - 1);

    return HI_SUCCESS;
}

HI_S32 BboxClip_N(HI_S32* ps32Proposals, HI_U32 u32ImageW, HI_U32 u32ImageH, HI_U32 u32Num)
{
    HI_S32 s32Ret = HI_FAILURE;
    for (HI_U32 i = 0; i < u32Num; i++)
    {
        s32Ret = BboxClip(&ps32Proposals[i*SVP_WK_PROPOSAL_WIDTH], u32ImageW, u32ImageH);
        SVP_FALSE_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);
    }
    return HI_SUCCESS;
}

/**************************************************
Function: BboxSmallSizeFilter
Description: remove the bboxes which are too small
***************************************************/
HI_S32 BboxSmallSizeFilter(HI_S32* ps32Proposals, HI_U32 u32minW, HI_U32 u32minH)
{
    HI_U32  u32ProposalW = (HI_U32)(ps32Proposals[2] - ps32Proposals[0] + 1);
    HI_U32  u32ProposalH = (HI_U32)(ps32Proposals[3] - ps32Proposals[1] + 1);

    if (u32ProposalW < u32minW || u32ProposalH < u32minH)
    {
        ps32Proposals[5] = RPN_SUPPRESS_TRUE;   // suppressed
    }

    return HI_SUCCESS;
}

HI_S32 BboxSmallSizeFilter_N(HI_S32* ps32Proposals, HI_U32 u32minW, HI_U32 u32minH, HI_U32 u32NumAnchors)
{
    HI_S32 s32Ret = HI_FAILURE;

    for (HI_U32 i = 0; i < u32NumAnchors; i++)
    {
        s32Ret = BboxSmallSizeFilter(&ps32Proposals[i*SVP_WK_PROPOSAL_WIDTH], u32minW, u32minH);
        SVP_FALSE_CHECK(HI_SUCCESS == s32Ret, HI_FAILURE);
    }
    return HI_SUCCESS;
}

/**************************************************
Function: dumpProposal
Description: dumpProposal info when DETECION_DBG
***************************************************/
HI_S32 dumpProposal(HI_S32* ps32Proposals, const HI_CHAR* filename, HI_U32 u32NumAnchors)
{
    if (DETECION_DBG)
    {
        FILE* file = fopen(filename, "w");
        SVP_FALSE_CHECK(NULL != file, HI_FAILURE);

        for (HI_U32 i = 0; i < u32NumAnchors; i++)
        {
            /*
            index     content
            0         x0
            1         y0
            2         x1
            3         y1
            4         score
            5         is suppressed or not
            */
            fprintf(file, "%f %d %d %d %d %d\n",
                (HI_FLOAT)ps32Proposals[4] / SVP_WK_QUANT_BASE,
                ps32Proposals[0],
                ps32Proposals[1],
                ps32Proposals[2],
                ps32Proposals[3],
                ps32Proposals[5]);

            ps32Proposals += SVP_WK_PROPOSAL_WIDTH;
        }

        fclose(file);
        file = NULL;
    }
    return HI_SUCCESS;
}

/**************************************************
Function: getRPNresult
Description: rite the final result to output
***************************************************/
HI_S32 getRPNresult(HI_S32* ps32ProposalResult, HI_U32* pu32NumRois, HI_U32 u32MaxRois,
    const HI_S32* ps32Proposals, HI_U32 u32NumAfterFilter)
{
    HI_U32 u32RoiCount = 0;
    for (HI_U32 i = 0; i < u32NumAfterFilter; i++)
    {
        if (RPN_SUPPRESS_FALSE == ps32Proposals[5])
        {
            // convert to 20.12 fixed-point number.
            ps32ProposalResult[SVP_WK_COORDI_NUM * u32RoiCount + 0] = ps32Proposals[0] * SVP_WK_QUANT_BASE;
            ps32ProposalResult[SVP_WK_COORDI_NUM * u32RoiCount + 1] = ps32Proposals[1] * SVP_WK_QUANT_BASE;
            ps32ProposalResult[SVP_WK_COORDI_NUM * u32RoiCount + 2] = ps32Proposals[2] * SVP_WK_QUANT_BASE;
            ps32ProposalResult[SVP_WK_COORDI_NUM * u32RoiCount + 3] = ps32Proposals[3] * SVP_WK_QUANT_BASE;

            u32RoiCount++;
        }
        ps32Proposals += SVP_WK_PROPOSAL_WIDTH;
        if (u32RoiCount >= u32MaxRois)
        {
            break;
        }
    }

    *pu32NumRois = u32RoiCount;
    /****************** write rpn_result *********************/
    if (DETECION_DBG)
    {
        FILE* rpn_result = fopen("rpn_result.txt", "w");
        SVP_FALSE_CHECK(NULL != rpn_result, HI_FAILURE);

        for (HI_U32 i = 0; i < u32RoiCount; i++)
        {
            fprintf(rpn_result, "%d %d %d %d\n",
                ps32ProposalResult[SVP_WK_COORDI_NUM*i + 0],
                ps32ProposalResult[SVP_WK_COORDI_NUM*i + 1],
                ps32ProposalResult[SVP_WK_COORDI_NUM*i + 2],
                ps32ProposalResult[SVP_WK_COORDI_NUM*i + 3]);
        }
        fclose(rpn_result);
        rpn_result = NULL;
    }

    return HI_SUCCESS;
}
