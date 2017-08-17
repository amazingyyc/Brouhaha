/**
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/8/9.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#if defined(real) && defined(BROU_NAME)

void BROU_NAME(matrixTranspose4X4)(real *src, size_t srcRowBytes, real *dst, size_t dstRowBytes) {
    real *realDst = dst;
    
    dst[0] = src[0]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[1]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[2]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[3];
    
    src = ((void*)src) + srcRowBytes;
    dst  = realDst + 1;
    
    dst[0] = src[0]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[1]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[2]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[3];
    
    src  =  ((void*)src) + srcRowBytes;
    dst  = realDst + 2;
    
    dst[0] = src[0]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[1]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[2]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[3];
    
    src = ((void*)src) + srcRowBytes;
    dst  = realDst + 3;
    
    dst[0] = src[0]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[1]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[2]; dst = ((void*)dst) + dstRowBytes;
    dst[0] = src[3];
}

/**
 * use neon to traspose matrix
 *
 * Not a good method!!
 * todo: add multi threads
 */
void BROU_NAME(brouTransposeMatrixBlockNeon)(real *in,
                                             size_t inRow, size_t inCol,
                                             real *out,
                                             size_t outRow, size_t outCol) {
    size_t inRowBytes  = sizeof(real) * inCol;
    size_t outRowBytes = sizeof(real) * outCol;
    
    for (size_t y = 0; y < inRow; y += 4) {
        y = BROU_MIN(inRow - 4, y);
        
        for (size_t x = 0; x < inCol; x += 4) {
            x = BROU_MIN(inCol - 4, x);
            
            BROU_NAME(matrixTranspose4X4Neon)(in + y * inCol + x, inRowBytes, out + x * outCol + y, outRowBytes);
        }
    }
}

/**
 * transpose 4X4 matrix every time
 * Not a good method!!
 * todo: add multi threads
 */
void BROU_NAME(brouTransposeMatrixBlock)(real *in,
                                         size_t inRow, size_t inCol,
                                         real *out,
                                         size_t outRow, size_t outCol) {
    size_t inRowBytes  = sizeof(real) * inCol;
    size_t outRowBytes = sizeof(real) * outCol;
    
    for (size_t y = 0; y < inRow; y += 4) {
        y = BROU_MIN(inRow - 4, y);
        
        for (size_t x = 0; x < inCol; x += 4) {
            x = BROU_MIN(inCol - 4, x);
            
            BROU_NAME(matrixTranspose4X4)(in + y * inCol + x, inRowBytes, out + x * outCol + y, outRowBytes);
        }
    }
}

/**
 * transpose the matrix use 2 loop
 */
void BROU_NAME(brouTransposeMatrixDirectly)(real *in,
                                            size_t inRow, size_t inCol,
                                            real *out,
                                            size_t outRow, size_t outCol) {
    for (size_t y = 0; y < inCol; ++y) {
        for (size_t x = 0; x < inRow; ++x) {
            out[y * outCol + x] = in[x * inCol + y];
        }
    }
}

/**
 * transpose the in matrix to out
 * outRow >= inCol outCol >= inRow
 */
void BROU_NAME(brouTransposeMatrix)(real *in, size_t inRow, size_t inCol, real *out, size_t outRow, size_t outCol) {
    if (4 > inRow || 4 > inCol) {
        BROU_NAME(brouTransposeMatrixDirectly)(in, inRow, inCol, out, outRow, outCol);
    } else {
#if defined(__ARM_NEON)
        BROU_NAME(brouTransposeMatrixBlockNeon)(in, inRow, inCol, out, outRow, outCol);
#elif
        BROU_NAME(brouTransposeMatrixBlock)(in, inRow, inCol, out, outRow, outCol);
#endif
    }
}

#endif






