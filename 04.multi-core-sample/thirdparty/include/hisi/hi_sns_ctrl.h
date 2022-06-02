/******************************************************************************

  Copyright (C), 2016, Hisilicon Tech. Co., Ltd.

 ******************************************************************************
  File Name     : hi_sns_ctrl.h
  Version       : Initial Draft
  Author        : Hisilicon multimedia software group
  Created       : 2011/01/10
  Description   :
  History       :
  1.Date        : 2011/01/10
    Author      :
    Modification: Created file

******************************************************************************/

#ifndef __HI_SNS_CTRL_H__
#define __HI_SNS_CTRL_H__

#include "hi_type.h"
#include "hi_comm_3a.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */


typedef struct hiISP_SNS_STATE_S
{
    HI_BOOL     bInit;                  /* HI_TRUE: Sensor init */
    HI_BOOL     bSyncInit;              /* HI_TRUE: Sync Reg init */
    HI_U8       u8ImgMode;
    HI_U8       u8Hdr;               /* HI_TRUE: HDR enbale */
    WDR_MODE_E  enWDRMode;

    ISP_SNS_REGS_INFO_S astRegsInfo[2]; /* [0]: Sensor reg info of cur-frame; [1]: Sensor reg info of pre-frame ; */

    HI_U32      au32FL[2];              /* [0]: FullLines of cur-frame; [1]: Pre FullLines of pre-frame */
    HI_U32      u32FLStd;               /* FullLines std */
    HI_U32      au32WDRIntTime[4];
} ISP_SNS_STATE_S;

typedef struct hiISP_SNS_OBJ_S
{
    HI_S32  (*pfnRegisterCallback)(VI_PIPE ViPipe, ALG_LIB_S *pstAeLib, ALG_LIB_S *pstAwbLib);
    HI_S32  (*pfnUnRegisterCallback)(VI_PIPE ViPipe, ALG_LIB_S *pstAeLib, ALG_LIB_S *pstAwbLib);
    HI_S32  (*pfnSetBusInfo)(VI_PIPE ViPipe, ISP_SNS_COMMBUS_U unSNSBusInfo);
    HI_VOID (*pfnStandby)(VI_PIPE ViPipe);
    HI_VOID (*pfnRestart)(VI_PIPE ViPipe);
    HI_S32  (*pfnWriteReg)(VI_PIPE ViPipe, HI_S32 s32Addr, HI_S32 s32Data);
    HI_S32  (*pfnReadReg)(VI_PIPE ViPipe, HI_S32 s32Addr);
    HI_S32  (*pfnSetInit)(VI_PIPE ViPipe, ISP_INIT_ATTR_S *pstInitAttr);
} ISP_SNS_OBJ_S;

extern ISP_SNS_OBJ_S stSnsMn34220Obj;
extern ISP_SNS_OBJ_S stSnsImx377Obj;
extern ISP_SNS_OBJ_S stSnsImx299Obj;
extern ISP_SNS_OBJ_S stSnsImx477Obj;
extern ISP_SNS_OBJ_S stSnsImx299SlvsObj;
extern ISP_SNS_OBJ_S stSnsImx290Obj;
extern ISP_SNS_OBJ_S stSnsImx334Obj;
extern ISP_SNS_OBJ_S stSnsOV2718Obj;
extern ISP_SNS_OBJ_S stSnsCmv50000Obj;
extern ISP_SNS_OBJ_S stSnsImx277SlvsObj;
extern ISP_SNS_OBJ_S stSnsImx117Obj;
extern ISP_SNS_OBJ_S stSnsImx290SlaveObj;
extern ISP_SNS_OBJ_S stSnsImx334SlaveObj;


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* __HI_SNS_CTRL_H__ */

