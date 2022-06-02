#ifndef __SAMPLE_SAVE_BLOB_H
#define __SAMPLE_SAVE_BLOB_H
#include "hi_runtime_api.h"
#ifdef ON_BOARD
#include "hi_ive.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif
HI_VOID saveBlob(const HI_CHAR* pszPath, const HI_RUNTIME_BLOB_S* pstSrcBlob, HI_U16 u16Index);
#ifdef __cplusplus
}
#endif

#endif
