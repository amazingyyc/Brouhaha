/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the ReLu operate
 */

#include <metal_stdlib>

#include "BrouStruct.metal"

using namespace metal;

/**
 * for 1d shape every thread output 1X4
 */
kernel void brouReLu1D(device half *input           [[buffer(0)]],
                       device half *output          [[buffer(1)]],
                       constant TensorShape& shape  [[buffer(2)]],
                       ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;
    
    if (index >= shape.dim0) {
        return;
    }
    
    device half4 *inputV  = (device half4*)(input  + index);
    device half4 *outputV = (device half4*)(output + index);
        
    outputV[0] = max(0, inputV[0]);
}

/**
 * every thread output 4X4 block
 * the width is timed by 4
 */
kernel void brouReLu2D(device half *input           [[buffer(0)]],
                       device half *output          [[buffer(1)]],
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

        device half4 *inputV  = (device half4*)(input  + offset);
        device half4 *outputV = (device half4*)(output + offset);

        outputV[0] = max(0, inputV[0]);
    }
}

/**
 * every thread output 4X4X4 block
 * the channel must be timed by 4
 */
kernel void brouReLu3D(device half *input           [[buffer(0)]],
                       device half *output          [[buffer(1)]],
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

            device half4 *inputV  = (device half4*)(input  + offset);
            device half4 *outputV = (device half4*)(output + offset);

            outputV[0] = max(0, inputV[0]);
        }
    }
}

