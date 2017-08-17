/**
 * BrouNeon.h
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/8/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#include "BrouNeon.h"

/**
 * traspose 4X4 half/uint16_t matrix use neon
 */
void matrixTranspose4X4Neon_uint16_t(uint16_t *src, size_t srcRowBytes, uint16_t *dst, size_t dstRowBytes) {
#if defined(__aarch64__)
    __asm__ __volatile__ (
                          "ld4  {v0.h, v1.h, v2.h, v3.h}[0], [%0], %1;        \n\t"
                          "ld4  {v0.h, v1.h, v2.h, v3.h}[1], [%0], %1;        \n\t"
                          "ld4  {v0.h, v1.h, v2.h, v3.h}[2], [%0], %1;        \n\t"
                          "ld4  {v0.h, v1.h, v2.h, v3.h}[3], [%0];            \n\t"

                          "st1 { v0.4h }, [%2], %3;                           \n\t"
                          "st1 { v1.4h }, [%2], %3;                           \n\t"
                          "st1 { v2.4h }, [%2], %3;                           \n\t"
                          "st1 { v3.4h }, [%2];                               \n\t"

                          :
                          :"r"(src), "r"(srcRowBytes), "r"(dst), "r"(dstRowBytes)
                          :"v0", "v1", "v2", "v3", "memory"
                          );
#elif defined(__ARM_NEON)
    __asm__ __volatile__ (
                          "vld4.16 {d0[0], d1[0], d2[0], d3[0]}, [%0], %1     \n\t"
                          "vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [%0], %1     \n\t"
                          "vld4.16 {d0[2], d1[2], d2[2], d3[2]}, [%0], %1     \n\t"
                          "vld4.16 {d0[3], d1[3], d2[3], d3[3]}, [%0]         \n\t"
                          
                          "vst1.16    d0, [%2], %3         \n\t"
                          "vst1.16    d1, [%2], %3         \n\t"
                          "vst1.16    d2, [%2], %3         \n\t"
                          "vst1.16    d3, [%2]             \n\t"
                          
                          :
                          :"r"(src), "r"(srcRowBytes), "r"(dst), "r"(dstRowBytes)
                          :"d0", "d1", "d2", "d3", "memory"
                          );
#endif
}

void matrixTranspose4X4Neon_half(uint16_t *src, size_t srcRowBytes, uint16_t *dst, size_t dstRowBytes) {
    matrixTranspose4X4Neon_uint16_t(src, srcRowBytes, dst, dstRowBytes);
}

/**
 * traspose 4X4 float matrix use neon
 */
void matrixTranspose4X4Neon_float(float *src, size_t srcRowBytes, float *dst, size_t dstRowBytes) {
#if defined(__aarch64__)
    __asm__ __volatile__ (
                          "ld4  {v0.s, v1.s, v2.s, v3.s}[0], [%0], %1;        \n\t"
                          "ld4  {v0.s, v1.s, v2.s, v3.s}[1], [%0], %1;        \n\t"
                          "ld4  {v0.s, v1.s, v2.s, v3.s}[2], [%0], %1;        \n\t"
                          "ld4  {v0.s, v1.s, v2.s, v3.s}[3], [%0];            \n\t"

                          "st1 { v0.4s }, [%2], %3;                           \n\t"
                          "st1 { v1.4s }, [%2], %3;                           \n\t"
                          "st1 { v2.4s }, [%2], %3;                           \n\t"
                          "st1 { v3.4s }, [%2];                               \n\t"

                          :
                          :"r"(src), "r"(srcRowBytes), "r"(dst), "r"(dstRowBytes)
                          :"v0", "v1", "v2", "v3", "memory"
                          );
#elif defined(__ARM_NEON)
    __asm__ __volatile__ (
                          "vld4.32 {d0[0], d2[0], d4[0], d6[0]}, [%0], %1     \n\t"
                          "vld4.32 {d0[1], d2[1], d4[1], d6[1]}, [%0], %1     \n\t"
                          "vld4.32 {d1[0], d3[0], d5[0], d7[0]}, [%0], %1     \n\t"
                          "vld4.32 {d1[1], d3[1], d5[1], d7[1]}, [%0]         \n\t"

                          "vst1.32    q0, [%2], %3         \n\t"
                          "vst1.32    q1, [%2], %3         \n\t"
                          "vst1.32    q2, [%2], %3         \n\t"
                          "vst1.32    q3, [%2]             \n\t"

                          :
                          :"r"(src), "r"(srcRowBytes), "r"(dst), "r"(dstRowBytes)
                          :"q0", "q1", "q2", "q3", "memory"
                          );
#endif
}

/**
 * double is just for test, the ios Metal does not support double
 */
void matrixTranspose4X4Neon_double(double *src, size_t srcRowBytes, double *dst, size_t dstRowBytes) {
#if defined(__aarch64__)
    __asm__ __volatile__ (
                          "ld4  {v0.d, v1.d, v2.d, v3.d}[0], [%0], %1         \n\t"
                          "ld4  {v0.d, v1.d, v2.d, v3.d}[1], [%0], %1         \n\t"
                          "ld4  {v4.d, v5.d, v6.d, v7.d}[0], [%0], %1         \n\t"
                          "ld4  {v4.d, v5.d, v6.d, v7.d}[1], [%0]             \n\t"

                          "st1 {v0.2d}, [%2], 16                              \n\t"
                          "st1 {v4.2d}, [%2]                                  \n\t"

                          "sub %2, %2, 16                                     \n\t"
                          "add %2, %2, %3                                     \n\t"

                          "st1 {v1.2d}, [%2], 16                              \n\t"
                          "st1 {v5.2d}, [%2]                                  \n\t"

                          "sub %2, %2, 16                                     \n\t"
                          "add %2, %2, %3                                     \n\t"

                          "st1 {v2.2d}, [%2], 16                              \n\t"
                          "st1 {v6.2d}, [%2]                                  \n\t"

                          "sub %2, %2, 16                                     \n\t"
                          "add %2, %2, %3                                     \n\t"

                          "st1 {v3.2d}, [%2], 16                              \n\t"
                          "st1 {v7.2d}, [%2]                                  \n\t"

                          :
                          :"r"(src), "r"(srcRowBytes), "r"(dst), "r"(dstRowBytes)
                          :"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
                          );
#endif
}





















