#include "sample_mutex_ops.h"


HI_S32 SAMPLE_MUTEX_Init(SAMPLE_MUTEX *pMutex)
{
#ifdef _WIN32
    InitializeCriticalSection(pMutex);
    return HI_SUCCESS;
#else
    return pthread_mutex_init(pMutex, NULL);
#endif
}

HI_S32 SAMPLE_MUTEX_Deinit(SAMPLE_MUTEX *pMutex)
{
#ifdef _WIN32
    DeleteCriticalSection(pMutex);
    return HI_SUCCESS;
#else
    return pthread_mutex_destroy(pMutex);
#endif
}

HI_S32 SAMPLE_MUTEX_Lock(SAMPLE_MUTEX *pMutex)
{
#ifdef _WIN32
    EnterCriticalSection(pMutex);
    return HI_SUCCESS;
#else
    return pthread_mutex_lock(pMutex);
#endif
}

HI_S32 SAMPLE_MUTEX_Unlock(SAMPLE_MUTEX *pMutex)
{
#ifdef _WIN32
    LeaveCriticalSection(pMutex);
    return HI_SUCCESS;
#else
    return pthread_mutex_unlock(pMutex);
#endif
}

HI_S32 SAMPLE_COND_Init(SAMPLE_COND *pCond)
{
#ifdef _WIN32
    InitializeConditionVariable(pCond);
    return HI_SUCCESS;
#else
    return pthread_cond_init(pCond, NULL);
#endif
}

HI_S32 SAMPLE_COND_Deinit(SAMPLE_COND *pCond)
{
#ifdef _WIN32
    return HI_SUCCESS;
#else
    return pthread_cond_destroy(pCond);
#endif
}

HI_S32 SAMPLE_COND_Signal(SAMPLE_COND *pCond)
{
#ifdef _WIN32
    WakeConditionVariable(pCond);
    return HI_SUCCESS;
#else
    return pthread_cond_signal(pCond);
#endif
}

HI_S32 SAMPLE_COND_Broadcast(SAMPLE_COND *pCond)
{
#ifdef _WIN32
    WakeAllConditionVariable(pCond);
    return HI_SUCCESS;
#else
    return pthread_cond_broadcast(pCond);
#endif
}

HI_S32 SAMPLE_COND_Wait(SAMPLE_COND *pCond, SAMPLE_MUTEX *pMutex)
{
#ifdef _WIN32
    SleepConditionVariableCS(pCond, pMutex, INFINITE);
    return HI_SUCCESS;
#else
    return pthread_cond_wait(pCond, pMutex);
#endif
}
