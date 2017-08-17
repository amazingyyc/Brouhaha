/**
 * BrouNeon.h
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/8/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#ifndef BrouNeon_h
#define BrouNeon_h

#include <stdio.h>
#include <arm_neon.h>

/**
 * transpose 4X4 matrix use neon
 * support half/uint16_t float double
 */
void matrixTranspose4X4Neon_half(uint16_t *src, size_t srcRowBytes, uint16_t *dst, size_t dstRowBytes);

void matrixTranspose4X4Neon_uint16_t(uint16_t *src, size_t srcRowBytes, uint16_t *dst, size_t dstRowBytes);

void matrixTranspose4X4Neon_float(float *src, size_t srcRowBytes, float *dst, size_t dstRowBytes);

#endif
