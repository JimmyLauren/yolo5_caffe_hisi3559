
/******************************************************************************
Copyright (C), 2016-2018, Hisilicon Tech. Co., Ltd.
******************************************************************************
File Name     : hi_comm_vpss.h
Version       : Initial Draft
Author        : Hisilicon multimedia software group
Created       : 2016/09/27
Last Modified :
Description   :
Function List :
******************************************************************************/

#ifndef __HI_COMM_VPSS_H__
#define __HI_COMM_VPSS_H__


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#include "hi_type.h"
#include "hi_common.h"
#include "hi_errno.h"
#include "hi_comm_video.h"

#define VPSS_SHARPEN_GAIN_NUM        (32)
#define VPSS_AUTO_ISO_STRENGTH_NUM   (16)
#define VPSS_YUV_SHPLUMA_NUM         (32)

#define HI_ERR_VPSS_NULL_PTR        HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_NULL_PTR)
#define HI_ERR_VPSS_NOTREADY        HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_SYS_NOTREADY)
#define HI_ERR_VPSS_INVALID_DEVID   HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_INVALID_DEVID)
#define HI_ERR_VPSS_INVALID_CHNID   HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_INVALID_CHNID)
#define HI_ERR_VPSS_EXIST           HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_EXIST)
#define HI_ERR_VPSS_UNEXIST         HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_UNEXIST)
#define HI_ERR_VPSS_NOT_SUPPORT     HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_SUPPORT)
#define HI_ERR_VPSS_NOT_PERM        HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_NOT_PERM)
#define HI_ERR_VPSS_NOMEM           HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_NOMEM)
#define HI_ERR_VPSS_NOBUF           HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_NOBUF)
#define HI_ERR_VPSS_ILLEGAL_PARAM   HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_ILLEGAL_PARAM)
#define HI_ERR_VPSS_BUSY            HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_BUSY)
#define HI_ERR_VPSS_BUF_EMPTY       HI_DEF_ERR(HI_ID_VPSS, EN_ERR_LEVEL_ERROR, EN_ERR_BUF_EMPTY)

#define VPSS_INVALID_FRMRATE   -1
#define VPSS_CHN0               0
#define VPSS_CHN1               1
#define VPSS_CHN2               2
#define VPSS_CHN3               3
#define VPSS_INVALID_CHN       -1

typedef enum hiNR_MOTION_MODE_E
{
    NR_MOTION_MODE_NORMAL     = 0,        /* normal */
    NR_MOTION_MODE_COMPENSATE = 1,        /* motion compensate */
    NR_MOTION_MODE_BUTT
}NR_MOTION_MODE_E;

typedef struct hiVPSS_NR_ATTR_S
{
    COMPRESS_MODE_E     enCompressMode;   /* RW; Reference frame compress mode */
    NR_MOTION_MODE_E    enNrMotionMode;   /* RW; NR motion compensate mode. */
}VPSS_NR_ATTR_S;

typedef struct hiVPSS_GRP_ATTR_S
{
    HI_U32                     u32MaxW;           /* RW; Range: Hi3559AV100 = [64, 16384] | Hi3556AV100 = [64, 8192]; Width of source image. */
    HI_U32                     u32MaxH;           /* RW; Range: Hi3559AV100 = [64, 16384] | Hi3556AV100 = [64, 8192]; Height of source image. */
    PIXEL_FORMAT_E             enPixelFormat;     /* RW; Pixel format of source image. */
    DYNAMIC_RANGE_E            enDynamicRange;    /* RW; DynamicRange of source image. */
    FRAME_RATE_CTRL_S          stFrameRate;       /* Grp frame rate contrl. */
    HI_BOOL                    bNrEn;             /* RW; NR enable. */
    VPSS_NR_ATTR_S             stNrAttr;          /* RW; NR attr. */
} VPSS_GRP_ATTR_S;

typedef enum hiVPSS_CHN_MODE_E
{
    VPSS_CHN_MODE_USER  = 0,       /* User mode. */
    VPSS_CHN_MODE_AUTO  = 1        /* Auto mode. */

} VPSS_CHN_MODE_E;

typedef struct hiVPSS_CHN_ATTR_S
{
    VPSS_CHN_MODE_E     enChnMode;          /* RW; Vpss channel's work mode. */
    HI_U32              u32Width;           /* RW; Range: Hi3559AV100 = [64, 16384] | Hi3556AV100 = [64, 8192]; Width of target image. */
    HI_U32              u32Height;          /* RW; Range: Hi3559AV100 = [64, 16384] | Hi3556AV100 = [64, 8192]; Height of target image. */
    VIDEO_FORMAT_E      enVideoFormat;      /* RW; Video format of target image. */
    PIXEL_FORMAT_E      enPixelFormat;      /* RW; Pixel format of target image. */
    DYNAMIC_RANGE_E     enDynamicRange;     /* RW; DynamicRange of target image. */
    COMPRESS_MODE_E     enCompressMode;     /* RW; Compression mode of the output. */
    FRAME_RATE_CTRL_S   stFrameRate;        /* Frame rate control info */
    HI_BOOL             bMirror;            /* RW; Mirror enable. */
    HI_BOOL             bFlip;              /* RW; Flip enable. */
    HI_U32              u32Depth;           /* RW; Range: [0, 8]; User get list depth. */
    ASPECT_RATIO_S      stAspectRatio;      /* Aspect Ratio info. */
} VPSS_CHN_ATTR_S;

typedef enum hiVPSS_CROP_COORDINATE_E
{
    VPSS_CROP_RATIO_COOR = 0,   /* Ratio coordinate. */
    VPSS_CROP_ABS_COOR          /* Absolute coordinate. */
} VPSS_CROP_COORDINATE_E;

typedef struct hiVPSS_CROP_INFO_S
{
    HI_BOOL                 bEnable;            /* RW; CROP enable. */
    VPSS_CROP_COORDINATE_E  enCropCoordinate;   /* RW; Coordinate mode of the crop start point. */
    RECT_S                  stCropRect;         /* CROP rectangular. */
} VPSS_CROP_INFO_S;

typedef struct hiVPSS_LDC_ATTR_S
{
    HI_BOOL     bEnable;                        /* RW;Whether LDC is enbale */
    LDC_ATTR_S  stAttr;
} VPSS_LDC_ATTR_S;

typedef struct hiVPSS_ROTATION_EX_ATTR_S
{
    HI_BOOL       bEnable;                      /* Whether ROTATE_EX_S is enbale */
    ROTATION_EX_S stRotationEx;                 /* Rotate Attribute */
}VPSS_ROTATION_EX_ATTR_S;

#define VPSS_PMFCOEF_NUM              (9UL)
typedef struct hiVPSS_PMF_ATTR_S
{
    HI_BOOL bEnable;                            /* RW;Whether PMF is enbale */
    SIZE_S  stDestSize;                         /* RW;Target size */
    HI_S64  as64PMFCoef[VPSS_PMFCOEF_NUM];      /* RW; Array of PMF coefficients */
} VPSS_PMF_ATTR_S;

typedef struct hiVPSS_LOW_DELAY_INFO_S
{
    HI_BOOL bEnable;          /* RW; Low delay enable. */
    HI_U32 u32LineCnt;        /* RW; Range: [16, 16384]; Low delay shoreline. */
}VPSS_LOW_DELAY_INFO_S;

typedef struct hiVPSS_EXT_CHN_ATTR_S
{
    VPSS_CHN           s32BindChn;      /* RW; Range: [0, 3]; Channel bind to. */
    HI_U32             u32Width;        /* RW; Range: Hi3559AV100 = [64, 16384] | Hi3556AV100 = [64, 8192]; Width of target image. */
    HI_U32             u32Height;       /* RW; Range: Hi3559AV100 = [64, 16384] | Hi3556AV100 = [64, 8192]; Height of target image. */
    VIDEO_FORMAT_E     enVideoFormat;   /* RW; Video format of target image. */
    PIXEL_FORMAT_E     enPixelFormat;   /* RW; Pixel format of target image. */
    DYNAMIC_RANGE_E    enDynamicRange;  /* RW; Dynamic range. */
    COMPRESS_MODE_E    enCompressMode;  /* RW; Compression mode of the output. */
    HI_U32             u32Depth;        /* RW; Range: [0, 8]; User get list depth. */
    FRAME_RATE_CTRL_S  stFrameRate;     /* Frame rate control info */
} VPSS_EXT_CHN_ATTR_S;



typedef struct hiVPSS_GRP_SHARPEN_MANUAL_ATTR_S
{
    HI_U16 au16TextureStr[VPSS_SHARPEN_GAIN_NUM];    /* RW; Range: [0, 4095]; Undirectional sharpen strength for texture and detail enhancement*/
    HI_U16 au16EdgeStr[VPSS_SHARPEN_GAIN_NUM];       /* RW; Range: [0, 4095]; Directional sharpen strength for edge enhancement*/
    HI_U16 u16TextureFreq;                           /* RW; Range: [0, 4095]; Texture frequency adjustment. Texture and detail will be finer when it increase*/
    HI_U16 u16EdgeFreq;                              /* RW; Range: [0, 4095]; Edge frequency adjustment. Edge will be narrower and thiner when it increase*/
    HI_U8  u8OverShoot;                              /* RW; Range: [0, 127]; u8OvershootAmt*/
    HI_U8  u8UnderShoot;                             /* RW; Range: [0, 127]; u8UndershootAmt*/
    HI_U8  u8ShootSupStr;                            /* RW; Range: [0, 255]; overshoot and undershoot suppression strength, the amplitude and width of shoot will be decrease when shootSupSt increase*/
    HI_U8  u8DetailCtrl;                             /* RW; Range: [0, 255]; Different sharpen strength for detail and edge. When it is bigger than 128, detail sharpen strength will be stronger than edge. */
} VPSS_GRP_SHARPEN_MANUAL_ATTR_S;

typedef struct hiVPSS_GRP_SHARPEN_AUTO_ATTR_S
{
    HI_U16 au16TextureStr[VPSS_SHARPEN_GAIN_NUM][VPSS_AUTO_ISO_STRENGTH_NUM]; /* RW; Range: [0, 4095]; Undirectional sharpen strength for texture and detail enhancement*/
    HI_U16 au16EdgeStr[VPSS_SHARPEN_GAIN_NUM][VPSS_AUTO_ISO_STRENGTH_NUM];    /* RW; Range: [0, 4095]; Directional sharpen strength for edge enhancement*/
    HI_U16 au16TextureFreq[VPSS_AUTO_ISO_STRENGTH_NUM];                       /* RW; Range: [0, 4095]; Texture frequency adjustment. Texture and detail will be finer when it increase*/
    HI_U16 au16EdgeFreq[VPSS_AUTO_ISO_STRENGTH_NUM];                          /* RW; Range: [0, 4095]; Edge frequency adjustment. Edge will be narrower and thiner when it increase*/
    HI_U8  au8OverShoot[VPSS_AUTO_ISO_STRENGTH_NUM];                          /* RW; Range: [0, 127]; u8OvershootAmt*/
    HI_U8  au8UnderShoot[VPSS_AUTO_ISO_STRENGTH_NUM];                         /* RW; Range: [0, 127]; u8UndershootAmt*/
    HI_U8  au8ShootSupStr[VPSS_AUTO_ISO_STRENGTH_NUM];                        /* RW; Range: [0, 255]; overshoot and undershoot suppression strength, the amplitude and width of shoot will be decrease when shootSupSt increase*/
    HI_U8  au8DetailCtrl[VPSS_AUTO_ISO_STRENGTH_NUM];                         /* RW; Range: [0, 255]; Different sharpen strength for detail and edge. When it is bigger than 128, detail sharpen strength will be stronger than edge. */
} VPSS_GRP_SHARPEN_AUTO_ATTR_S;


typedef struct hiVPSS_GRP_SHARPEN_ATTR_S
{
    HI_BOOL                         bEnable;                          /* RW; Sharpen enable. */
    OPERATION_MODE_E                enOpType;                         /* RW; Sharpen Operation mode. */
    HI_U8                           au8LumaWgt[VPSS_YUV_SHPLUMA_NUM]; /* RW; Range: [0, 127]; sharpen weight based on loacal luma*/
    VPSS_GRP_SHARPEN_MANUAL_ATTR_S  stSharpenManualAttr;              /* RW; Sharpen manual attribute*/
    VPSS_GRP_SHARPEN_AUTO_ATTR_S    stSharpenAutoAttr;                /* RW; Sharpen auto attribute*/
} VPSS_GRP_SHARPEN_ATTR_S;

/****************************VPSS 3DNR********************/

/* 3DNR X interface for Hi3556AV100 */
typedef struct
{
    HI_U8  IES0, IES1, IES2, IES3;
    HI_U16 IEDZ : 10,  _rb_ : 6;
} tV56aIEy;

typedef struct
{
    HI_U8  SPN6 : 3, SFR  : 5;
    HI_U8  SBN6 : 3, PBR6 : 5;

    HI_U8  SFS2, SFT2, SBR2;
    HI_U8  SFS4, SFT4, SBR4;

    HI_U16 STH1 : 9,  SFN1 : 3, SFN0  : 3, NRyEn : 1;
    HI_U16 STH2 : 9,  SFN2 : 3, BWSF4 : 1, kMode : 3;
    HI_U16 STH3 : 9,  SFN3 : 3, tEdge : 2, TriTh : 1, _rb_  : 1;
} tV56aSFy;

typedef struct
{
    HI_U16 MADZ : 10, MAI0 : 2, MAI1 : 2,  MAI2 : 2;
    HI_U8  MADK,      MABR;

    HI_U16 MATH : 10, MATE : 4, MATW : 2;
    HI_U8  MASW :  4, MABW : 3, MAXN : 1, _rB_;
} tV56aMDy;

typedef struct
{
    HI_U16 TFS : 4,  TDZ : 10, TDX : 2;
    HI_U8  TFR[5],   TSS : 4,  TSI : 1, _rb_ : 2;
    HI_U16 SDZ : 10, STR : 5,  bRef : 1;
} tV56aTFy;

typedef struct
{
    HI_U8  SFC, _rb_ : 2, TFC : 6;

    HI_U16 CSFS : 10,    CSFR : 6;
    HI_U16 CTFS :  4,    CIIR : 1;
    HI_U16 CTFR : 11;
} tV56aNRc;

typedef struct
{
    tV56aIEy IEy[2];
    tV56aSFy SFy[4];
    tV56aMDy MDy[2];
    tV56aTFy TFy[2];
    tV56aNRc NRc;

    HI_U16 SBSk2[32], SDSk2[32];
    HI_U16 SBSk3[32], SDSk3[32];
} VPSS_NRX_PARAM_V1_S;

/* 3DNR interface */
typedef enum hiVPSS_NR_VER_E
{
    VPSS_NR_V1 = 1,
    VPSS_NR_V2 = 2,
    VPSS_NR_V3 = 3,
    VPSS_NR_V4 = 4,
    VPSS_NR_BUTT
}VPSS_NR_VER_E;

typedef struct hiVPSS_GRP_NRX_PARAM_S
{
    VPSS_NR_VER_E enNRVer;
    union
    {
        VPSS_NRX_PARAM_V1_S stNRXParam_V1;   /* interface X V1 for Hi3556AV100 */
    };

}VPSS_GRP_NRX_PARAM_S;


typedef struct hiVPSS_PARAM_MOD_S
{
    HI_BOOL bOneBufForLowDelay;
    HI_U32  u32VpssVbSource;
}VPSS_MOD_PARAM_S;


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */
#endif /* __HI_COMM_VPSS_H__ */

