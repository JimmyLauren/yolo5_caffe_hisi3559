#ifndef __SAMPLE_RUNTIME_DEFINE_H
#define __SAMPLE_RUNTIME_DEFINE_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef ON_BOARD
#define RESOURCE_DIR "./resource"
#else
#define RESOURCE_DIR "../resource"
#endif

#define MAX_ROI_NUM 300

#if DEBUG
    #define sample_debug(...) \
    do \
{ \
printf(__VA_ARGS__); \
} while (0)
#else
#define sample_debug(...)
#endif

#ifdef __cplusplus
}
#endif

#endif
