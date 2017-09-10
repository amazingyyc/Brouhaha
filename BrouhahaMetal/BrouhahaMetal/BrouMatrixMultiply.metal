/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * matrix multiply
 */
#if defined(real) && defined(real4) && defined(BROU)

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
kernel void BROU(MatrixMultiply)(device real *A     [[buffer(0)]],
                                 device real *B     [[buffer(1)]],
                                 device real *C     [[buffer(2)]],
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
    
    device real4 *aV = (device real4*)(A + row);
    device real4 *bV = (device real4*)(B + col);
    
    real4 a, b;
    real4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    
    int loopCount = k;
    
    do {
        a = aV[0];
        b = bV[0];
        
        c0 += a.x * b;
        c1 += a.y * b;
        c2 += a.z * b;
        c3 += a.w * b;
        
        aV = (device real4*)((device real*)aV + m);
        bV = (device real4*)((device real*)bV + n);
    } while(--loopCount);
    
    device real4 *cV = (device real4*)(C + row * n + col);
    
    cV[0] = c0; cV = (device real4*)((device real*)cV + n);
    cV[0] = c1; cV = (device real4*)((device real*)cV + n);
    cV[0] = c2; cV = (device real4*)((device real*)cV + n);
    cV[0] = c3;
}

/**
 * the output c will add bias
 * the bias dimension is (shape.dim2)
 */
kernel void BROU(MatrixMultiplyWithBias)(device real *A                 [[buffer(0)]],
                                         device real *B                 [[buffer(1)]],
                                         device real *C                 [[buffer(2)]],
                                         device real *bia               [[buffer(3)]],
                                         constant TensorShape& shape    [[buffer(4)]],
                                         ushort2 grid [[thread_position_in_grid]]) {
    int m = shape.dim0;
    int k = shape.dim1;
    int n = shape.dim2;
    
    int row = grid.y << 2;
    int col = grid.x << 2;
    
    if (row >= m || col >= n) {
        return;
    }
    
    device real4 *aV = (device real4*)(A + row);
    device real4 *bV = (device real4*)(B + col);
    
    real4 a, b;
    real4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    
    int loopCount = k;
    
    do {
        a = aV[0];
        b = bV[0];
        
        c0 += a.x * b;
        c1 += a.y * b;
        c2 += a.z * b;
        c3 += a.w * b;
        
        aV = (device real4*)((device real*)aV + m);
        bV = (device real4*)((device real*)bV + n);
    } while(--loopCount);
    
    real4 biaV = ((device real4*)(bia + col))[0];
    device real4 *cV = (device real4*)(C + row * n + col);
    
    cV[0] = c0 + biaV; cV = (device real4*)((device real*)cV + n);
    cV[0] = c1 + biaV; cV = (device real4*)((device real*)cV + n);
    cV[0] = c2 + biaV; cV = (device real4*)((device real*)cV + n);
    cV[0] = c3 + biaV;
}

#endif











