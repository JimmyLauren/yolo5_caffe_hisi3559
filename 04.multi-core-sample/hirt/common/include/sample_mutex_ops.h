#ifndef SAMPLE_MUTEX_OPS_H
#define SAMPLE_MUTEX_OPS_H
#include "back/hi_type.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION SAMPLE_MUTEX;
typedef CONDITION_VARIABLE SAMPLE_COND;
#else
#include <pthread.h>
typedef pthread_mutex_t SAMPLE_MUTEX;
typedef pthread_cond_t SAMPLE_COND;
#endif

HI_S32 SAMPLE_MUTEX_Init(SAMPLE_MUTEX *pMutex);
HI_S32 SAMPLE_MUTEX_Deinit(SAMPLE_MUTEX *pMutex);
HI_S32 SAMPLE_MUTEX_Lock(SAMPLE_MUTEX *pMutex);
HI_S32 SAMPLE_MUTEX_Unlock(SAMPLE_MUTEX *pMutex);

HI_S32 SAMPLE_COND_Init(SAMPLE_COND *pCond);
HI_S32 SAMPLE_COND_Deinit(SAMPLE_COND *pCond);
HI_S32 SAMPLE_COND_Signal(SAMPLE_COND *pCond);
HI_S32 SAMPLE_COND_Broadcast(SAMPLE_COND *pCond);
HI_S32 SAMPLE_COND_Wait(SAMPLE_COND *pCond, SAMPLE_MUTEX *pMutex);

#ifdef __cplusplus
}
#endif

#endif
