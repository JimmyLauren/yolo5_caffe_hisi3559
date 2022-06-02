#include <stdio.h>
#include "sample_save_blob.h"
#ifdef _WIN32
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <linux/limits.h>
#endif
#ifdef ON_BOARD
#include "sample_ive_image.h"
#else
#include "sample_cv_saveblob.h"
#endif

HI_VOID saveBlob(const HI_CHAR* pszPath, const HI_RUNTIME_BLOB_S* pstSrcBlob, HI_U16 u16Index)
{
    HI_CHAR aszFileName[PATH_MAX] = {0};
#ifdef ON_BOARD
    snprintf(aszFileName, sizeof(aszFileName), "%s.ppm", pszPath);
    saveBlobByIVE(aszFileName, pstSrcBlob, u16Index);
#else
    snprintf(aszFileName, sizeof(aszFileName), "%s.png", pszPath);
    saveBlobByCV(aszFileName, pstSrcBlob, u16Index);
#endif
}
