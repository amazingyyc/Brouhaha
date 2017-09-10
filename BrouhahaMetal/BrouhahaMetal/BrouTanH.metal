/**
 * BrouhahaMetal
 *
 * Created by yanyuanchi on 2017/8/14.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the TanH operate,
 */

#if defined(real) && defined(real4) && defined(BROU)

/**
 * for 1d shape every thread output 1X4
 */
kernel void BROU(TanH1D)(device real *input           [[buffer(0)]],
                         device real *output          [[buffer(1)]],
                         constant TensorShape& shape  [[buffer(2)]],
                         ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;
    
    if (index >= shape.dim0) {
        return;
    }
    
    device real4 *inputV  = (device real4*)(input  + index);
    device real4 *outputV = (device real4*)(output + index);
    
    outputV[0] = tanh(inputV[0]);
}

/**
 * every thread output 4X4 block
 * the width is timed by 4
 */
kernel void BROU(TanH2D)(device real *input           [[buffer(0)]],
                         device real *output          [[buffer(1)]],
                         constant TensorShape& shape  [[buffer(2)]],
                         ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    
    if (y >= height || x >= width) {
        return;
    }
    
    int maxJ = min(y + 4, height);
    
    for (int j = y; j < maxJ; ++j) {
        int offset = j * width + x;
        
        device real4 *inputV  = (device real4*)(input  + offset);
        device real4 *outputV = (device real4*)(output + offset);
        
        outputV[0] = tanh(inputV[0]);
    }
}

/**
 * every thread output 4X4X4 block
 * the channel must be timed by 4
 */
kernel void BROU(TanH3D)(device real *input           [[buffer(0)]],
                         device real *output          [[buffer(1)]],
                         constant TensorShape& shape  [[buffer(2)]],
                         ushort3 grid [[thread_position_in_grid]]) {
    int height  = shape.dim0;
    int width   = shape.dim1;
    int channel = shape.dim2;
    
    int y = grid.y << 2;
    int x = grid.x << 2;
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
            
            outputV[0] = tanh(inputV[0]);
        }
    }
}

#endif









