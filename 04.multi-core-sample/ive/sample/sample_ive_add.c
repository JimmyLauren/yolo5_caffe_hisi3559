//
// Created by liujian on 1/12/22.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <semaphore.h>
#include <pthread.h>
#include <math.h>
#include <back/hi_type.h>
#include "sample_comm_ive.h"


HI_VOID SAMPLE_IVE_Add(HI_VOID)
{
    IVE_IMAGE_S a;

    /* 系统资源初始化 */
    SAMPLE_COMM_IVE_CheckIveMpiInit();

    /* 相关变量定义&&内存申请 */
    printf("SAMPLE_IVE_Add sample.\n");
    IVE_HANDLE hIveHandle = 0;
    HI_BOOL bInstant = HI_TRUE;
    HI_S32  s32Ret = HI_FAILURE;
    IVE_SRC_IMAGE_S srcImage1 = {0};
    IVE_SRC_IMAGE_S srcImage2 = {0};
    IVE_DST_IMAGE_S addImage = {0};

    IVE_DST_IMAGE_S dstTmp1 = {0};
    IVE_DST_IMAGE_S dstTmp2 = {0};

    IVE_ADD_CTRL_S stAddCtrl;
    stAddCtrl.u0q16X = 32768;
    stAddCtrl.u0q16Y = 32768;

    SAMPLE_COMM_IVE_CreateImage(&srcImage1,IVE_IMAGE_TYPE_U8C1,64,64);
    SAMPLE_COMM_IVE_CreateImage(&srcImage2,IVE_IMAGE_TYPE_U8C1,64,64);
    SAMPLE_COMM_IVE_CreateImage(&addImage,IVE_IMAGE_TYPE_U8C1,64,64);

    SAMPLE_COMM_IVE_CreateImage(&dstTmp1,IVE_IMAGE_TYPE_U8C1,64,64);
    SAMPLE_COMM_IVE_CreateImage(&dstTmp2,IVE_IMAGE_TYPE_U8C1,64,64);

    /* set values for variables */
    memset ((HI_VOID*)srcImage1.au64VirAddr[0],1,srcImage1.au32Stride[0] * srcImage1.u32Height);
//    memset ((HI_VOID*)srcImage1.au64VirAddr[0]+(HI_U64)5,5,(HI_U64)1);

    memset ((HI_VOID*)srcImage2.au64VirAddr[0],3,srcImage2.au32Stride[0] * srcImage2.u32Height);
//    memset ((HI_VOID*)srcImage2.au64VirAddr[0]+(HI_U64)5,8,(HI_U64)1);

    memset ((HI_VOID*)addImage.au64VirAddr[0],9,addImage.au32Stride[0] * addImage.u32Height);

    memset ((HI_VOID*)dstTmp1.au64VirAddr[0],100,dstTmp1.au32Stride[0] * dstTmp1.u32Height);
    memset ((HI_VOID*)dstTmp2.au64VirAddr[0],100,dstTmp2.au32Stride[0] * dstTmp2.u32Height);

#if 1
    printf("srcImage1:\n");
    for(unsigned int i=0;i<srcImage1.u32Width*srcImage1.u32Height;i++)
    {
        HI_U8 * p = (HI_U8*)srcImage1.au64VirAddr[0];
        printf("%d ",p[i]);
    }
    printf("\n");

    printf("srcImage2:\n");
    for(unsigned int i=0;i<srcImage2.u32Width*srcImage2.u32Height;i++)
    {
        HI_U8 * p = (HI_U8*)srcImage2.au64VirAddr[0];
        printf("%d ",p[i]);
    }
    printf("\n");
#endif




    // 中值滤波，过滤像素值波动影响
    IVE_ORD_STAT_FILTER_CTRL_S midFlt;
    midFlt.enMode = IVE_ORD_STAT_FILTER_MODE_MEDIAN;

    IVE_DST_IMAGE_S middleOutput={0};
    SAMPLE_COMM_IVE_CreateImage(&middleOutput,IVE_IMAGE_TYPE_U8C1,64,64);
    memset((HI_VOID*)middleOutput.au64VirAddr[0],100,middleOutput.au32Stride[0] * middleOutput.u32Height);

    s32Ret = HI_MPI_IVE_OrdStatFilter(&hIveHandle,&srcImage1,&middleOutput,&midFlt,bInstant);
    if(s32Ret != 0)
    {
        printf("HI_MPI_IVE_OrdStatFilter failed. Err code:%#x.\n",s32Ret);
    }
#if 0
    printf("middleOutput:\n");
    for(unsigned int i=0;i<middleOutput.u32Width*middleOutput.u32Height;i++)
    {
        HI_U8 * p = (HI_U8*)middleOutput.au64VirAddr[0];
        printf("%d ",p[i]);
    }
    printf("\n");
#endif
     //*/

    // 使用filter实现乘法功能
    IVE_FILTER_CTRL_S stFilterCtrl =
        {
            {
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 2, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            }, 0
        };

    s32Ret = HI_MPI_IVE_Filter(&hIveHandle,&srcImage1,&dstTmp1,&stFilterCtrl,bInstant);
    if(s32Ret != 0)
    {
        printf("HI_MPI_IVE_Filter failed. Err code:%#x.\n",s32Ret);
    }

    s32Ret = HI_MPI_IVE_Filter(&hIveHandle,&srcImage2,&dstTmp2,&stFilterCtrl,bInstant);
    if(s32Ret != 0)
    {
        printf("HI_MPI_IVE_Filter failed. Err code:%#x.\n",s32Ret);
    }


    // 实现两图像相减
    IVE_SUB_CTRL_S stSubCtrl;
    stSubCtrl.enMode = IVE_SUB_MODE_ABS;

    IVE_DST_IMAGE_S dstSubImage;
    SAMPLE_COMM_IVE_CreateImage(&dstSubImage,IVE_IMAGE_TYPE_U8C1,64,64);
    s32Ret = HI_MPI_IVE_Sub(&hIveHandle, &srcImage1, &srcImage2, &dstSubImage, &stSubCtrl, bInstant);
    if(s32Ret!=0)
    {
        printf("HI_MPI_IVE_Sub failed. Err code:%#x.\n",s32Ret);
    }
#if 1
    printf("dstSubImage:\n");
    for(unsigned int i=0;i<dstSubImage.u32Width*dstSubImage.u32Height;i++)
    {
        HI_U8 * p = (HI_U8*)dstSubImage.au64VirAddr[0];
        printf("%d ",p[i]);
    }
    printf("\n");
#endif



    /* 实现两图像相加操作 */
    s32Ret = HI_MPI_IVE_Add(&hIveHandle,&dstTmp1,&dstTmp2,&addImage,&stAddCtrl,bInstant);
    if(s32Ret != 0)
    {
        printf("HI_MPI_IVE_Add failed. Err code:%#x.\n",s32Ret);
    }


#if 1
    printf("addImage:\n");
    for(unsigned int i=0;i<addImage.u32Width*addImage.u32Height;i++)
    {
        HI_U8 * p = (HI_U8*)addImage.au64VirAddr[0];
        printf("%d ",p[i]);
    }
    printf("\n");
#endif


    /* 相关变量&&系统资源释放 */
    IVE_MMZ_FREE(srcImage1.au64PhyAddr[0],srcImage1.au64VirAddr[0]);
    IVE_MMZ_FREE(srcImage2.au64PhyAddr[0],srcImage2.au64VirAddr[0]);
    IVE_MMZ_FREE(addImage.au64PhyAddr[0],addImage.au64VirAddr[0]);

    IVE_MMZ_FREE(dstTmp1.au64PhyAddr[0],dstTmp1.au64VirAddr[0]);
    IVE_MMZ_FREE(dstTmp2.au64PhyAddr[0],dstTmp2.au64VirAddr[0]);
//    IVE_MMZ_FREE(middleOutput.au64PhyAddr[0],middleOutput.au64VirAddr[0]);
    SAMPLE_COMM_IVE_IveMpiExit();
}




