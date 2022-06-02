#include "hi_common.h"
#include "hi_comm_video.h"

#ifndef SAMPLE_PRT
#define SAMPLE_PRT(fmt...)   \
do {\
    printf("[%s]-%d: ", __FUNCTION__, __LINE__);\
    printf(fmt);\
}while(0)
#endif


typedef enum hiTHREAD_CONTRL_E
{
    THREAD_CTRL_START,
    THREAD_CTRL_PAUSE,
    THREAD_CTRL_STOP,
}THREAD_CONTRL_E;

typedef struct hiVDEC_THREAD_PARAM_S
{
    HI_S32 s32ChnId;
    PAYLOAD_TYPE_E enType;
    HI_CHAR cFilePath[128];
    HI_CHAR cFileName[128];
    HI_S32 s32StreamMode;
    HI_S32 s32MilliSec;
    HI_S32 s32MinBufSize;
    HI_S32 s32IntervalTime;
    THREAD_CONTRL_E eThreadCtrl;
    HI_U64  u64PtsInit;
    HI_U64  u64PtsIncrease;
    HI_BOOL bCircleSend;
}VDEC_THREAD_PARAM_S;


HI_S32 SAMPLE_COMM_VDEC_DatafifoInit(HI_S32 s32VdecChnNum);
HI_VOID SAMPLE_COMM_VDEC_DatafifoDeinit(HI_S32 s32VdecChnNum);
HI_VOID SAMPLE_COMM_VDEC_StartSendStream(HI_S32 s32ChnNum, VDEC_THREAD_PARAM_S *pstVdecSend, pthread_t *pVdecThread);
HI_VOID SAMPLE_COMM_VDEC_StopSendStream(HI_S32 s32ChnNum, VDEC_THREAD_PARAM_S *pstVdecSend, pthread_t *pVdecThread);
HI_VOID SAMPLE_COMM_VDEC_CmdCtrl(HI_S32 s32ChnNum,VDEC_THREAD_PARAM_S *pstVdecSend, pthread_t *pVdecThread);
