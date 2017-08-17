/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * just test the matrix mutiply
 */
#include <metal_stdlib>
#include <metal_compute>

using namespace metal;

/**
 * C = A * B
 * Where C is M x N, A is M x K, and B is K x N.
 */
constant int M[[function_constant(0)]];
constant int K[[function_constant(1)]];
constant int N[[function_constant(2)]];

/**
 * the A matrix a row-major
 * the B is col-major
 */
kernel void brouMatrixMultiplyFloat32(device float *A[[buffer(0)]],
                                      device float *B[[buffer(1)]],
                                      device float *C[[buffer(2)]],
                                      uint2 grid [[thread_position_in_grid]]) {
    if (grid.x >= N || grid.y >= M) {
        return;
    }

    device float *aData = A + grid.y * K;
    device float *bData = B + grid.x * K;

    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;

    int limit = K - 3;
    int i = 0;

    for (; i < limit; i += 4) {
        sum0 += aData[i] * bData[i];
        sum1 += aData[i + 1] * bData[i + 1];
        sum2 += aData[i + 2] * bData[i + 2];
        sum3 += aData[i + 3] * bData[i + 3];
    }

    for (; i < K; ++i) {
        sum0 += aData[i] * bData[i];
    }

    C[grid.y * N + grid.x] = sum0 + sum1 + sum2 + sum3;
}

struct Matrix {
    int width;
    int height;
    int stride;

    device float* data;
};

/**
 * get a sub-matrix of m
 * data have been splited to many blocks and the block'size is (blockSize, blockSize)
 */
inline Matrix getSubMatrix(device float *data, int stride, int row, int col, int blockSize) {
    Matrix subMatrix;

    subMatrix.width  = blockSize;
    subMatrix.height = blockSize;
    subMatrix.stride = stride;
    subMatrix.data   = data + stride * row * blockSize + col * blockSize;

    return subMatrix;
}

inline float getData(const Matrix m, int row, int col) {
    return m.data[row * m.stride + col];
}

constexpr constant int blockSize = 16;

/**
 * ref:
 * Matrix Multiplication with CUDA — A basic introduction to the CUDA programming model
 * Robert Hochberg
 */
kernel void brouMatrixMultiplyFloat32ShareMemory(device float *A[[buffer(0)]],
                                                 device float *B[[buffer(1)]],
                                                 device float *C[[buffer(2)]],
                                                 uint2 threadInGroup [[thread_position_in_threadgroup]],
                                                 uint2 groupInGrid   [[threadgroup_position_in_grid]],
                                                 uint2 threadInGrid  [[thread_position_in_grid]]) {
    int threadRow = threadInGrid.y;
    int threadCol = threadInGrid.x;

    /**the current block's index*/
    int blockRow = groupInGrid.y;
    int blockCol = groupInGrid.x;

    int row = threadInGroup.y;
    int col = threadInGroup.x;

    /**store the accumulate the value of result*/
    float cValue = 0;

    int loopCount = K / blockSize;
    int lastSize  = K - loopCount * blockSize;

    threadgroup float aShared[blockSize][blockSize];
    threadgroup float bShared[blockSize][blockSize];

    for (int i = 0; i < loopCount; ++i) {
        Matrix aSubMatrix = getSubMatrix(A, K, blockRow, i, blockSize);
        Matrix bSubMatrix = getSubMatrix(B, N, i, blockCol, blockSize);

        if (threadRow < M && threadCol < N) {
            aShared[row][col] = getData(aSubMatrix, row, col);
            bShared[row][col] = getData(bSubMatrix, row, col);
        } else if (threadRow < M) {
            aShared[row][col] = getData(aSubMatrix, row, col);
        } else if (threadCol < N) {
            bShared[row][col] = getData(bSubMatrix, row, col);
        }

        threadgroup_barrier(mem_flags::mem_none);

        /**clculate the result*/
        if (threadRow < M && threadCol < N) {
            for (int e = 0; e < blockSize; ++e) {
                cValue += aShared[row][e] * bShared[e][col];
            }
        }

        threadgroup_barrier(mem_flags::mem_none);
    }

    if (0 != lastSize) {
        Matrix aSubMatrix = getSubMatrix(A, K, blockRow, loopCount, blockSize);
        Matrix bSubMatrix = getSubMatrix(B, N, loopCount, blockCol, blockSize);

        if (threadRow < M && threadCol < N) {
            if (col < lastSize) {
                aShared[row][col] = getData(aSubMatrix, row, col);
            }

            if (row < lastSize) {
                bShared[row][col] = getData(bSubMatrix, row, col);
            }
        } else if (threadRow < M) {
            if (col < lastSize) {
                aShared[row][col] = getData(aSubMatrix, row, col);
            }
        } else if (threadCol < N) {
            if (row < lastSize) {
                bShared[row][col] = getData(bSubMatrix, row, col);
            }
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (threadRow < M && threadCol < N) {
            for (int e = 0; e < lastSize; ++e) {
                cValue += aShared[row][e] * bShared[e][col];
            }
        }
    }

    if (threadRow < M && threadCol < N) {
        C[threadRow * N + threadCol] = cValue;
    }
}

/**
 * ref:MatrixMultiplicationPerformanceTest
 * every block will deal with a block which's size is 8x8
 * the A matrix's size is (m, n) and col-major, the pointer A's memory has been padded to (n, _M) (the _M timed by 8)
 * the B matrix's size is (n, k) and row-major, the pointer B's memory bad benn padded to (n, _K) (the _K timed by 8)
 * the C'size is (_M, _K) row-major
 */

/**
 * aRowByte = _M * sizeof(float)
 * bRowByte = _N * sizeof(float)
 */
constant int aRowByte[[function_constant(3)]];
constant int bRowByte[[function_constant(4)]];

kernel void brouMatrixMultiplyFloat32SpliteBlock(device float *A[[buffer(0)]],
                                                 device float *B[[buffer(1)]],
                                                 device float *C[[buffer(2)]],
                                                 uint2 grid  [[thread_position_in_grid]]) {
    int row = grid.y << 3;
    int col = grid.x << 3;

    if (row >= M || col >= N) {
        return;
    }

    device float4 *aV = (device float4*)(A + row);
    device float4 *bV = (device float4*)(B + col);
    device float4 *cV = (device float4*)((device float*)((device char*)C + row * bRowByte) + col);

    /**
     * use 16 float4 to store the 8x8 result
     */
    float4 c0 = 0.f, c1 = 0.f,  c2 = 0.f,  c3 = 0.f,  c4 = 0.f,  c5 = 0.f,  c6 = 0.f,  c7 = 0.f;
    float4 c8 = 0.f, c9 = 0.f, c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f, c14 = 0.f, c15 = 0.f;

    int loopCount = K;

    do {
        float4 a0 = aV[0];
        float4 a1 = aV[1];
        float4 b0 = bV[0];
        float4 b1 = bV[1];

        c0 += a0.x * b0;
        c1 += a0.x * b1;
        c2 += a0.y * b0;
        c3 += a0.y * b1;
        c4 += a0.z * b0;
        c5 += a0.z * b1;
        c6 += a0.w * b0;
        c7 += a0.w * b1;

        c8  += a1.x * b0;
        c9  += a1.x * b1;
        c10 += a1.y * b0;
        c11 += a1.y * b1;
        c12 += a1.z * b0;
        c13 += a1.z * b1;
        c14 += a1.w * b0;
        c15 += a1.w * b1;

        aV = (device float4*)((device char*)aV + aRowByte);
        bV = (device float4*)((device char*)bV + bRowByte);
    } while(--loopCount);

    cV[0] = c0; cV[1] = c1;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c2; cV[1] = c3;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c4; cV[1] = c5;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c6; cV[1] = c7;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c8; cV[1] = c9;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c10; cV[1] = c11;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c12; cV[1] = c13;
    cV = (device float4*)((device char*)cV + bRowByte);

    cV[0] = c14; cV[1] = c15;
}

/**
 * ref:MatrixMultiplicationPerformanceTest
 * every block will deal with a block which's size is 4X4
 * the A matrix's size is (m, n) and col-major, the pointer A's memory has been padded to (n, _M) (the _M timed by 4)
 * the B matrix's size is (n, k) and row-major, the pointer B's memory bad benn padded to (n, _K) (the _K timed by 4)
 * the C'size is (_M, _K) row-major
 *
 * aRowByte = _M * sizeof(float)
 * bRowByte = _N * sizeof(float)
 */
kernel void brouMatrixMultiplyFloat32SpliteBlockX4(device float *A[[buffer(0)]],
                                                   device float *B[[buffer(1)]],
                                                   device float *C[[buffer(2)]],
                                                   uint2 grid  [[thread_position_in_grid]]) {
    int row = grid.y << 2;
    int col = grid.x << 2;

    if (row >= M || col >= N) {
        return;
    }

    device float4 *aV = (device float4*)(A + row);
    device float4 *bV = (device float4*)(B + col);
    device float4 *cV = (device float4*)((device float*)((device char*)C + row * bRowByte) + col);

    float4 a, b;
    float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    int loopCount  = K;

    do {
        a = aV[0];
        b = bV[0];

        c0 += a.x * b;
        c1 += a.y * b;
        c2 += a.z * b;
        c3 += a.w * b;

        aV = (device float4*)((device char*)aV + aRowByte);
        bV = (device float4*)((device char*)bV + bRowByte);
    } while(--loopCount);

    cV[0] = c0; cV = (device float4*)((device char*)cV + bRowByte);
    cV[0] = c1; cV = (device float4*)((device char*)cV + bRowByte);
    cV[0] = c2; cV = (device float4*)((device char*)cV + bRowByte);
    cV[0] = c3;
}















