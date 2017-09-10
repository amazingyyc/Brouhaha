#if defined(type) && defined(real) && defined(BROU)

void BROU(MatrixTranspose4X4)(type *src, size_t srcRowBytes, type *dst, size_t dstRowBytes) {
    type *realDst = dst;
    
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
 * transpose 4X4 matrix every time
 * todo: add multi threads
 */
void BROU(TransposeMatrixBlock)(type *in, size_t inRow, size_t inCol, type *out, size_t outRow, size_t outCol) {
    size_t inRowBytes  = sizeof(type) * inCol;
    size_t outRowBytes = sizeof(type) * outCol;
    
    for (size_t y = 0; y < inRow; y += 4) {
        y = BROU_MIN(inRow - 4, y);
        
        for (size_t x = 0; x < inCol; x += 4) {
            x = BROU_MIN(inCol - 4, x);
            
#if defined(__ARM_NEON)
            BROU(MatrixTranspose4X4Neon)(in + y * inCol + x, inRowBytes, out + x * outCol + y, outRowBytes);
#else
            BROU(MatrixTranspose4X4)(in + y * inCol + x, inRowBytes, out + x * outCol + y, outRowBytes);
#endif
        }
    }
}

/**
 * transpose the matrix use 2 loop
 */
void BROU(TransposeMatrixDirectly)(type *in, size_t inRow, size_t inCol, type *out, size_t outRow, size_t outCol) {
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
void BROU(TransposeMatrix)(type *in, size_t inRow, size_t inCol, type *out, size_t outRow, size_t outCol) {
    if (4 > inRow || 4 > inCol) {
        BROU(TransposeMatrixDirectly)(in, inRow, inCol, out, outRow, outCol);
    } else {
        BROU(TransposeMatrixBlock)(in, inRow, inCol, out, outRow, outCol);
    }
}

#endif






