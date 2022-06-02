#ifndef __SAMPLE_MEMORY_OPS_H
#define __SAMPLE_MEMORY_OPS_H
#include "back/hi_type.h"
#include "hi_runtime_comm.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef SAMPLE_MEM_DELETE
#define SAMPLE_MEM_DELETE(ptr) \
    do \
    { \
        if ((ptr)) \
        { \
            delete (ptr); \
            ptr = HI_NULL; \
        } \
    } while (0)
#endif

#ifndef SAMPLE_MEM_DELETE_ARRAY
#define SAMPLE_MEM_DELETE_ARRAY(p)\
    do{\
        if(p){\
            delete [](p);\
            (p) = HI_NULL;\
        }\
    } while (0)
#endif

#ifndef SAMPLE_FREE
#define SAMPLE_FREE(ptr) \
    do \
    { \
        if ((ptr)) \
        { \
            free(ptr); \
            (ptr) = HI_NULL; \
        } \
    } while (0)
#endif

HI_S32 SAMPLE_AllocMem(HI_RUNTIME_MEM_S *pstMemInfo, HI_BOOL bCached);
HI_S32 SAMPLE_FlushCache(HI_RUNTIME_MEM_S *pstMemInfo);
HI_S32 SAMPLE_FreeMem(HI_RUNTIME_MEM_S *pstMemInfo);

#ifdef __cplusplus
}
#endif

#endif
