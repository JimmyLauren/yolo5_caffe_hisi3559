#ifndef __SAMPLE_RESIZE_ROI_H
#define __SAMPLE_RESIZE_ROI_H
#include "hi_runtime_api.h"
#include "sample_data_utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

HI_S32 resizeROI(const HI_RUNTIME_BLOB_S* pstSrcBlob,
                 const HI_S32 as32Coord[],
                 HI_U32 u32CoordStride,
                 const HI_U16 u16RoiCnt,
                 HI_RUNTIME_BLOB_S* pstDstBlob,
                 const HI_U16 u16DstOffSetCnt);

HI_S32 resizeBlob(const HI_RUNTIME_BLOB_S* pstSrcBlob,
                  HI_RUNTIME_BLOB_S* pstDstBlob);

HI_S32 cropPadBlob(const HI_RUNTIME_BLOB_S* pstSrcBlob,
    const HI_RUNTIME_BLOB_S* pstBboxBlob,
    HI_RUNTIME_BLOB_S* pstDstBlob,
    TRACKER_LOCATION_S* pstLocation,
    HI_S32* ps32EdgeX,
    HI_S32* ps32EdgeY,
    HI_S32* ps32RegionW,
    HI_S32* ps32RegionH);

HI_VOID drawImageRect(const HI_CHAR* pszPicPath, const HI_RUNTIME_BLOB_S* pstBlob,
                         HI_S32 as32Coord[], HI_U32 u32CoordCnt, HI_U32 u32CoordStride);
#ifdef __cplusplus
}
#endif
#endif
