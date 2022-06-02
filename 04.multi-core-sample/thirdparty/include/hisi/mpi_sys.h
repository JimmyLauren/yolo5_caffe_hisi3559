
/******************************************************************************
Copyright (C), 2016, Hisilicon Tech. Co., Ltd.
******************************************************************************
File Name     : mpi_sys.h
Version       : Initial Draft
Author        : Hisilicon multimedia software group
Created       : 2016/07/15
Last Modified :
Description   : MPP Programe Interface for system control moudle
Function List :
******************************************************************************/


#ifndef __MPI_SYS_H__
#define __MPI_SYS_H__

#include "hi_type.h"
#include "hi_common.h"
#include "hi_comm_sys.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

HI_S32 HI_MPI_SYS_Init(HI_VOID);
HI_S32 HI_MPI_SYS_Exit(HI_VOID);

HI_S32 HI_MPI_SYS_SetConfig(const MPP_SYS_CONFIG_S* pstSysConfig);
HI_S32 HI_MPI_SYS_GetConfig(MPP_SYS_CONFIG_S* pstSysConfig);

HI_S32  HI_MPI_SYS_Bind(MPP_CHN_S* pstSrcChn, MPP_CHN_S* pstDestChn);
HI_S32  HI_MPI_SYS_UnBind(MPP_CHN_S* pstSrcChn, MPP_CHN_S* pstDestChn);
HI_S32  HI_MPI_SYS_GetBindbyDest(MPP_CHN_S* pstDestChn, MPP_CHN_S* pstSrcChn);
HI_S32  HI_MPI_SYS_GetBindbySrc(MPP_CHN_S *pstSrcChn, MPP_BIND_DEST_S *pstBindDest);

HI_S32 HI_MPI_SYS_GetVersion(MPP_VERSION_S* pstVersion);

/*
** u64Base is the global PTS of the system.
** ADVICE:
** 1. Better to call HI_MPI_SYS_GetCurPTS on the host board to get u64Base.
** 2. When os start up, call HI_MPI_SYS_InitPTSBase to set the init PTS.
** 3. When media bussines is running, synchronize the PTS one time per minute
**     by calling HI_MPI_SYS_SyncPTS.
*/
HI_S32 HI_MPI_SYS_GetCurPTS(HI_U64* pu64CurPTS);
HI_S32 HI_MPI_SYS_InitPTSBase(HI_U64 u64PTSBase);
HI_S32 HI_MPI_SYS_SyncPTS(HI_U64 u64PTSBase);

/* alloc mmz memory in user context                                         */
HI_S32 HI_MPI_SYS_MmzAlloc(HI_U64* pu64PhyAddr, HI_VOID** ppVirAddr,
                           const HI_CHAR* strMmb, const HI_CHAR* strZone, HI_U32 u32Len);

/* alloc mmz memory with cache */
HI_S32 HI_MPI_SYS_MmzAlloc_Cached(HI_U64* pu64PhyAddr, HI_VOID** ppVirAddr,
                                  const HI_CHAR* pstrMmb, const HI_CHAR* pstrZone, HI_U32 u32Len);

/* free mmz memory in user context                                          */
HI_S32 HI_MPI_SYS_MmzFree(HI_U64 u64PhyAddr, HI_VOID* pVirAddr);

/* fulsh cache */
HI_S32 HI_MPI_SYS_MmzFlushCache(HI_U64 u64PhyAddr, HI_VOID* pVirAddr, HI_U32 u32Size);

/*
** Call the mmap function to map physical address to virtual address
** The system function mmap is too complicated, so we packge it.
*/
HI_VOID* HI_MPI_SYS_Mmap(HI_U64 u64PhyAddr, HI_U32 u32Size);
HI_VOID* HI_MPI_SYS_MmapCache(HI_U64 u64PhyAddr, HI_U32 u32Size);
HI_S32 HI_MPI_SYS_Munmap(HI_VOID* pVirAddr, HI_U32 u32Size);
HI_S32 HI_MPI_SYS_MflushCache(HI_U64 u64PhyAddr, HI_VOID *pVirAddr, HI_U32 u32Size);

HI_S32 HI_MPI_SYS_SetMemConfig(MPP_CHN_S* pstMppChn, const HI_CHAR* pcMmzName);
HI_S32 HI_MPI_SYS_GetMemConfig(MPP_CHN_S* pstMppChn, HI_CHAR* pcMmzName);

/* Close all the FD which is used by sys module */
HI_S32 HI_MPI_SYS_CloseFd(HI_VOID);


/* Get virtual meminfo according to virtual addr, should be in one process */
HI_S32 HI_MPI_SYS_GetVirMemInfo(const void* pVirAddr, SYS_VIRMEM_INFO_S* pstMemInfo);

/* Set/get Scale coefficient level for VPSS/VGS*/
HI_S32 HI_MPI_SYS_SetScaleCoefLevel(SCALE_RANGE_S *pstScaleRange,SCALE_COEFF_LEVEL_S *pstScaleCoeffLevel);
HI_S32 HI_MPI_SYS_GetScaleCoefLevel(SCALE_RANGE_S *pstScaleRange,SCALE_COEFF_LEVEL_S *pstScaleCoeffLevel);

/* Set/Get local timezone, range: [-86400, 86400] seconds (that is: [-24, 24] hours)  */
HI_S32 HI_MPI_SYS_SetTimeZone(HI_S32 s32TimeZone);
HI_S32 HI_MPI_SYS_GetTimeZone(HI_S32 *ps32TimeZone);

HI_S32 HI_MPI_SYS_SetGPSInfo(GPS_INFO_S *pstGPSInfo);
HI_S32 HI_MPI_SYS_GetGPSInfo(GPS_INFO_S *pstGPSInfo);

HI_S32 HI_MPI_SYS_SetTuningConnect(HI_S32 s32Connect);
HI_S32 HI_MPI_SYS_GetTuningConnect(HI_S32* ps32Connect);


HI_S32 HI_MPI_SYS_SetVIVPSSMode(VI_VPSS_MODE_S* pstVIVPSSMode);
HI_S32 HI_MPI_SYS_GetVIVPSSMode(VI_VPSS_MODE_S* pstVIVPSSMode);


#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /*__MPI_SYS_H__ */

