/**
 * BrouConvolutionConvert.c
 * Brouhaha

 * Created by yanyuanchi on 2017/6/13.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * this is just for test
 */
#include <memory.h>

#include "arm_neon.h"

/**
 * transpose 4X4 matrix
 */
void matrixTranspose4X4(uint16_t *src, int srcWidth, uint16_t *dst, int dstWidth) {
    uint16_t *realDst = dst;

    dst[0] = src[0]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[1]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[2]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[3];

    src = ((void*)src) + srcWidth;
    dst  = realDst + 1;

    dst[0] = src[0]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[1]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[2]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[3];

    src  =  ((void*)src) + srcWidth;
    dst  = realDst + 2;

    dst[0] = src[0]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[1]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[2]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[3];

    src = ((void*)src) + srcWidth;
    dst  = realDst + 3;

    dst[0] = src[0]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[1]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[2]; dst = ((void*)dst) + dstWidth;
    dst[0] = src[3];
}

/**
 * use the neon to transpose a 4X4 matrix
 */
void matrixTranspose4X4NeonIns(void *src, int srcWidth, void *dst, int dstWidth) {
    uint16x4x4_t in;

    in = vld4_lane_u16(src, in, 0); src += srcWidth;
    in = vld4_lane_u16(src, in, 1); src += srcWidth;
    in = vld4_lane_u16(src, in, 2); src += srcWidth;
    in = vld4_lane_u16(src, in, 3);

    vst1_u16(dst, in.val[0]);  dst += dstWidth;
    vst1_u16(dst, in.val[1]);  dst += dstWidth;
    vst1_u16(dst, in.val[2]);  dst += dstWidth;
    vst1_u16(dst, in.val[3]);
}

/**
 * srcwidth the bytes of src's row
 * dstWidth the bytes of dst's row
 */
void matrixTranspose4X4NeonAsm(void *src, int srcWidth, void *dst, int dstWidth) {
#if defined(__aarch64__)
    __asm__ __volatile__ (
                  "ld4  {v0.h, v1.h, v2.h, v3.h}[0], [%0], x1        \n\t"
                  "ld4  {v0.h, v1.h, v2.h, v3.h}[1], [%0], x1        \n\t"
                  "ld4  {v0.h, v1.h, v2.h, v3.h}[2], [%0], x1        \n\t"
                  "ld4  {v0.h, v1.h, v2.h, v3.h}[3], [%0]            \n\t"

                  "st1 { v0.4h }, [%1], x3                           \n\t"
                  "st1 { v1.4h }, [%1], x3                           \n\t"
                  "st1 { v2.4h }, [%1], x3                           \n\t"
                  "st1 { v3.4h }, [%1]                               \n\t"

                  :
                  :"r"(src), "r"(dst)
                  :"v0", "v1", "v2", "v3", "memory"
    );
#elif defined(__ARM_NEON)
    __asm__ __volatile__ (
                  "vld4.16 {d0[0], d1[0], d2[0], d3[0]}, [%0], r1     \n\t"
                  "vld4.16 {d0[1], d1[1], d2[1], d3[1]}, [%0], r1     \n\t"
                  "vld4.16 {d0[2], d1[2], d2[2], d3[2]}, [%0], r1     \n\t"
                  "vld4.16 {d0[3], d1[3], d2[3], d3[3]}, [%0]         \n\t"

                  "vst1.16    d0, [%1], r3         \n\t"
                  "vst1.16    d1, [%1], r3         \n\t"
                  "vst1.16    d2, [%1], r3         \n\t"
                  "vst1.16    d3, [%1]             \n\t"

                  :
                  :"r"(src), "r"(dst)
                  :"d0", "d1", "d2", "d3", "memory"
                  );
#else
    matrixTranspose4X4(src, srcWidth, dst, dstWidth);
#endif
}

/**
 * kernel's dimension is (outputChannel, kernelHeight, kernelWidth, inputChannel)
 * the matrix's dimension is (kernelHeight*kernelWidth*inputChannel, outputChannelX4)
 * the kernel can be seen(outputChannel, kernelHeight*kernelWidth*inputChannel)
 * transposition the kernel and stored in matrix
 */
void convert4DKernelToMatrix(uint16_t *kernel, uint16_t *matrix,
                             int outputChannel, int kernelHeight, int kernelWidth, int inputChannel,
                             int outputChannelX4) {
    int M = outputChannel;
    int N = kernelHeight * kernelWidth * inputChannel;
    int MX4 = outputChannelX4;
    

    /**set 0*/
    int zeroByteSize = (MX4 - M) * 2;
    uint16_t *zeroOffset = matrix + M;

    int kernelRowBytes = N * 2;
    int matrixRowBytes = MX4 * 2;

    for (int y = 0; y < N; ++y) {
        memset(zeroOffset, 0, zeroByteSize);
        zeroOffset += MX4;
    }
    
    
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < M; ++x) {
            matrix[y * MX4 + x] = kernel[x * N + y];
        }
    }
    
    return;

    for (int y = 0; y < M; y += 4) {
        y = y > (M - 4) ? (M - 4) : y;

        for (int x = 0; x < N; x += 4) {
            x = x > (N - 4) ? (N - 4) : x;
            
            matrixTranspose4X4(kernel + y * N + x, kernelRowBytes, matrix + x * MX4 + y, matrixRowBytes);
        }
    }
}











