#include <stdio.h>
#include <stdlib.h>
#include "hi_runtime_api.h"
#include "string.h"
#include "sample_memory_ops.h"
#ifdef ON_BOARD
#include "mpi_sys.h"
#include "mpi_vb.h"
#else
#include "back/hi_comm_svp.h"
#include "back/hi_nnie.h"
#include "back/mpi_nnie.h"
#endif
#include "math.h"
#include "sample_log.h"
#include "sample_save_blob.h"
#include "sample_resize_roi.h"
#include "sample_data_utils.h"
#include "sample_runtime_classify.h"
#include "sample_runtime_detection_rfcn.h"
#include "sample_runtime_group_rfcnalex.h"
#include "sample_runtime_detection_ssd.h"
#include "sample_runtime_group_tracker.h"

#if DEBUG
#define sample_debug(...) \
    do \
    { \
        printf(__VA_ARGS__); \
    } while (0)
#else
#define sample_debug(...)
#endif


static void SAMPLE_RUNTIME_Usage(char* pchPrgName)
{
    printf("Usage : %s <index> \n", pchPrgName);
    printf("index:\n");
    printf("\t 0) Alexnet(VNode)\n");
    printf("\t 1) RFCN(VNode->RNode->VNode)\n");
    printf("\t 2) RFCN & AlexNet Group(RFCN->Connector->AlexNet)\n");
    printf("\t 3) SSD\n");
    printf("\t 4) RFCN & Goturn & Alexnet\n");
}

#ifdef ON_BOARD
static HI_S32 SAMPLE_COMM_SVP_SysInit(HI_VOID)
{
    HI_S32 s32Ret = HI_FAILURE;
    VB_CONFIG_S struVbConf;

    HI_MPI_SYS_Exit();
    HI_MPI_VB_Exit();

    memset(&struVbConf, 0, sizeof(VB_CONFIG_S));

    struVbConf.u32MaxPoolCnt             = 2;
    struVbConf.astCommPool[1].u64BlkSize = 768 * 576 * 2;
    struVbConf.astCommPool[1].u32BlkCnt  = 1;

    s32Ret = HI_MPI_VB_SetConfig((const VB_CONFIG_S*)&struVbConf);

    if (HI_SUCCESS != s32Ret)
    {
        printf("HI_MPI_VB_SetConfig error\n");
        return HI_FAILURE;
    }

    s32Ret = HI_MPI_VB_Init();

    if (HI_SUCCESS != s32Ret)
    {
        printf("HI_MPI_VB_Init error\n");
        return HI_FAILURE;
    }

    s32Ret = HI_MPI_SYS_Init();

    if (HI_SUCCESS != s32Ret)
    {
        printf("HI_MPI_SYS_Init error\n");
        return HI_FAILURE;
    }

    return s32Ret;
}

static HI_S32 SAMPLE_COMM_SVP_SysExit(HI_VOID)
{
    HI_S32 s32Ret = HI_FAILURE;

    s32Ret = HI_MPI_SYS_Exit();

    if (HI_SUCCESS != s32Ret)
    {
        printf("HI_MPI_SYS_Exit error\n");
        return HI_FAILURE;
    }

    s32Ret = HI_MPI_VB_Exit();

    if (HI_SUCCESS != s32Ret)
    {
        printf("HI_MPI_VB_Exit error\n");
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}
#endif

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        SAMPLE_RUNTIME_Usage(argv[0]);
        return HI_FAILURE;
    }

#ifdef ON_BOARD
    SAMPLE_COMM_SVP_SysInit();
#endif

    switch (*argv[1])
    {
        case '0':
            SAMPLE_AlexNet();
            break;
        case '1':
            SAMPLE_RFCN();
            break;
        case '2':
            SAMPLE_Model_Group_RFCNAlexNet();
            break;
        case '3':
            SAMPLE_SSD();
            break;
        case '4':
            SAMPLE_Model_Group_RFCN_GOTURN_ALEXNET(1, 5);
            break;
        default:
            printf("index[%s] error !!!!!!!!!!\n", argv[1]);
            SAMPLE_RUNTIME_Usage(argv[0]);
            break;
    }

#ifdef ON_BOARD
    SAMPLE_COMM_SVP_SysExit();
#endif
    return HI_SUCCESS;
}
