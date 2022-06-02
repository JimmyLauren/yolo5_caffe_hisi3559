#ifndef SAMPLE_LOG_H
#define SAMPLE_LOG_H
#include <stdio.h>
#include "back/hi_type.h"

#ifdef _WIN32
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#else
#include <linux/limits.h>
#endif

#ifndef SAMPLE_LOG_PRINT
#define SAMPLE_LOG_PRINT printf
#endif

#ifndef SAMPLE_CHK_GOTO
#define SAMPLE_CHK_GOTO(val, label, ...) \
    do \
    { \
        if ((val)) \
        { \
            SAMPLE_LOG_PRINT(__VA_ARGS__); \
            goto label; \
        } \
    } while (0)
#endif

#ifndef SAMPLE_CHK_PRINTF
#define SAMPLE_CHK_PRINTF(val, ...) \
    do \
    { \
        if ((val)) \
        { \
            SAMPLE_LOG_PRINT(__VA_ARGS__); \
        } \
    } while (0)
#endif

#ifndef SAMPLE_CHK_RETURN
#define SAMPLE_CHK_RETURN(val, ret, ...) \
    do \
    { \
        if ((val)) \
        { \
            SAMPLE_LOG_PRINT(__VA_ARGS__); \
            return (ret); \
        } \
    } while (0)
#endif

#ifndef SAMPLE_CHK_RETURN_VOID
#define SAMPLE_CHK_RETURN_VOID(val, ...) \
    do \
    { \
        if ((val)) \
        { \
            SAMPLE_LOG_PRINT(__VA_ARGS__); \
            return; \
        } \
    } while (0)
#endif

#ifndef SAMPLE_CHK_RETURN_NO_PRINT
#define SAMPLE_CHK_RETURN_NO_PRINT(val, ret) \
    do \
    { \
        if ((val)) \
        { \
            return (ret); \
        } \
    } while (0)
#endif

#endif
