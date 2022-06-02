#ifndef __SAMPLE_IVE_IMAGE_H
#define __SAMPLE_IVE_IMAGE_H
#ifdef ON_BOARD
#include "hi_type.h"
#include "hi_runtime_comm.h"
#include "sample_data_utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

HI_S32 cropPadBlobByIVE(const HI_RUNTIME_BLOB_S* pstSrcBlob,
    const HI_RUNTIME_BLOB_S* pstBboxBlob,
    HI_RUNTIME_BLOB_S* pstDstBlob,
    TRACKER_LOCATION_S* pstLocation,
    HI_S32* ps32EdgeX,
    HI_S32* ps32EdgeY,
    HI_S32* ps32RegionW,
    HI_S32* ps32RegionH);

HI_S32 resizeByIVE(const HI_RUNTIME_BLOB_S *pstSrcBlob, HI_RUNTIME_BLOB_S* pstDstBlob);

HI_VOID drawImageRectByIVE(const HI_CHAR* pszPicPath, const HI_RUNTIME_BLOB_S* pstBlob,
                         HI_S32 as32Coord[], HI_U32 u32CoordCnt, HI_U32 u32CoordStride);
HI_VOID saveBlobByIVE(const HI_CHAR* pszPath, const HI_RUNTIME_BLOB_S* pstSrcBlob, HI_U16 u16Index);

#ifdef __cplusplus
}
#endif
#endif
#endif
