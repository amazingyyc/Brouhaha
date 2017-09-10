/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the linear operate
 */
#if defined(real) && defined(real4) && defined(BROU)

/**
 * f(x) = a * x + b
 */
kernel void BROU(Linear1D)(device real *input          [[buffer(0)]],
                           device real *output         [[buffer(1)]],
                           device real *ab             [[buffer(2)]],
                           constant TensorShape& shape [[buffer(3)]],
                           ushort grid [[thread_position_in_grid]]) {
    int len = shape.dim0;
    
    int index = grid << 2;
    
    if (index >= len) {
        return;
    }
    
    device real4 *inputV  = (device real4*)(input  + index);
    device real4 *outputV = (device real4*)(output + index);
    
    outputV[0] = ab[0] * inputV[0] + ab[1];
}

kernel void BROU(Linear2D)(device real *input          [[buffer(0)]],
                           device real *output         [[buffer(1)]],
                           device real *ab             [[buffer(2)]],
                           constant TensorShape& shape [[buffer(3)]],
                           ushort2 grid [[thread_position_in_grid]]) {
    /**the width must be timed by 4*/
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int maxJ = min(y + 4, height);
    
    for (int j = y; j < maxJ; ++j) {
        device real4 *inputV  = (device real4*)(input  + j * width + x);
        device real4 *outputV = (device real4*)(output + j * width + x);
        
        outputV[0] = ab[0] * inputV[0] + ab[1];
    }
}

kernel void BROU(Linear3D)(device real *input          [[buffer(0)]],
                           device real *output         [[buffer(1)]],
                           device real *ab             [[buffer(2)]],
                           constant TensorShape& shape [[buffer(3)]],
                           ushort3 grid [[thread_position_in_grid]]) {
    /**the channel must be timed by 4*/
    int height  = shape.dim0;
    int width   = shape.dim1;
    int channel = shape.dim2;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (y >= height || x >= width || z >= channel) {
        return;
    }
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            int offset = (j * width + i) * channel + z;
            
            device real4 *inputV  = (device real4*)(input  + offset);
            device real4 *outputV = (device real4*)(output + offset);
            
            outputV[0] = ab[0] * inputV[0] + ab[1];
        }
    }
}

#endif











