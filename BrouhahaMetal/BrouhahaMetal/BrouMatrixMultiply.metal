/**
 * handle matrix multiply for convolution
 */

#include <metal_stdlib>

#include "BrouStruct.metal"

using namespace metal;

/**
 * A's dimension is (kernelHeight*kernelWidth*inputChannel, [outputHeight*outputWidth]X4)
 * B's dimension is (kernelHeight*kernelWidth*inputChannel, outputChannelX4)
 * C's dimension is ([outputHeight*outputWidth]X4, outputChannelX4)
 * the bias's dimenson is (outputChannelX4)
 *
 * the output'd dimension is ([outputHeight*outputWidth]X4, outputChannelX4)
 * so it constains the real output data (outputHeight*outputWidth, outputChannelX4)
 */

/**
 * M = outputHeight*outputWidth
 * K = kernelHeight*kernelWidth*inputChannel
 * N = outputChannel
 *
 * MX4 is the smallest number that bigger than M and time of 4
 * NX4 is the smallest number that bigger than N and time of 4
 */
constant int M[[function_constant(0)]];
constant int K[[function_constant(1)]];
constant int N[[function_constant(2)]];

constant int MX4[[function_constant(3)]];
constant int NX4[[function_constant(4)]];

kernel void brouMatrixMultiply(device half *A   [[buffer(0)]],
                               device half *B   [[buffer(1)]],
                               device half *bia [[buffer(2)]],
                               device half *C   [[buffer(3)]],
                               ushort2 grid [[thread_position_in_grid]]) {
    int row = grid.y << 2;
    int col = grid.x << 2;

    if (row >= M || col >= N) {
        return;
    }

    device half4 *aV = (device half4*)(A + row);
    device half4 *bV = (device half4*)(B + col);

    half4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half4 a, b;

    int loopCount = K;

    do {
        a = aV[0];
        b = bV[0];

        c0 += a.x * b;
        c1 += a.y * b;
        c2 += a.z * b;
        c3 += a.w * b;

        aV = (device half4*)((device half*)aV + MX4);
        bV = (device half4*)((device half*)bV + NX4);
    } while(--loopCount);

    device half4 *cV = (device half4*)(C + row * NX4 + col);
    half4 biaV = ((device half4*)(bia + col))[0];

    cV[0] = c0 + biaV; cV = (device half4*)((device half*)cV + NX4);
    cV[0] = c1 + biaV; cV = (device half4*)((device half*)cV + NX4);
    cV[0] = c2 + biaV; cV = (device half4*)((device half*)cV + NX4);
    cV[0] = c3 + biaV;
}

kernel void brouMatrixMultiplyWithoutBias(device half *A   [[buffer(0)]],
                                          device half *B   [[buffer(1)]],
                                          device half *C   [[buffer(2)]],
                                          ushort2 grid [[thread_position_in_grid]]) {
    int row = grid.y << 2;
    int col = grid.x << 2;

    if (row >= M || col >= N) {
        return;
    }

    device half4 *aV = (device half4*)(A + row);
    device half4 *bV = (device half4*)(B + col);

    half4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    half4 a, b;

    int loopCount = K;

    do {
        a = aV[0];
        b = bV[0];

        c0 += a.x * b;
        c1 += a.y * b;
        c2 += a.z * b;
        c3 += a.w * b;

        aV = (device half4*)((device half*)aV + MX4);
        bV = (device half4*)((device half*)bV + NX4);
    } while(--loopCount);

    device half4 *cV = (device half4*)(C + row * NX4 + col);

    cV[0] = c0; cV = (device half4*)((device half*)cV + NX4);
    cV[0] = c1; cV = (device half4*)((device half*)cV + NX4);
    cV[0] = c2; cV = (device half4*)((device half*)cV + NX4);
    cV[0] = c3;
}

/**
 * A dimension is (shape.dim0, shape.dim1)
 * B dimesnion is (shape.dim1, shape.dim2)
 * C dimension is (shape.dim0, shape.dim2)
 * the shape.dim0 and shape.dim2 must be timed by 4
 *
 * the A is col-major
 * the B is row-major
 * the C is row-major
 */
kernel void brouMatrixMultiplyWithShape(device half *A     [[buffer(0)]],
                                        device half *B     [[buffer(1)]],
                                        device half *C     [[buffer(2)]],
                                        constant TensorShape& shape [[buffer(3)]],
                                        ushort2 grid [[thread_position_in_grid]]) {
    int m = shape.dim0;
    int k = shape.dim1;
    int n = shape.dim2;
    
    int row = grid.y << 2;
    int col = grid.x << 2;
    
    if (row >= m || col >= n) {
        return;
    }
    
    device half4 *aV = (device half4*)(A + row);
    device half4 *bV = (device half4*)(B + col);
    
    half4 a, b;
    half4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    
    int loopCount = k;
    
    do {
        a = aV[0];
        b = bV[0];
        
        c0 += a.x * b;
        c1 += a.y * b;
        c2 += a.z * b;
        c3 += a.w * b;
        
        aV = (device half4*)((device half*)aV + m);
        bV = (device half4*)((device half*)bV + n);
    } while(--loopCount);
    
    device half4 *cV = (device half4*)(C + row * n + col);
    
    cV[0] = c0; cV = (device half4*)((device half*)cV + n);
    cV[0] = c1; cV = (device half4*)((device half*)cV + n);
    cV[0] = c2; cV = (device half4*)((device half*)cV + n);
    cV[0] = c3;
}













