/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the linear operate
 */

#include <metal_stdlib>
#include <metal_math>

#include "BrouStruct.metal"

using namespace metal;

/**
 * f(x) = a * x + b
 */
typedef struct {
    half a;
    half b;
} AB;

kernel void brouLinear1D(device half *input          [[buffer(0)]],
                         device half *output         [[buffer(1)]],
                         constant AB& ab             [[buffer(2)]],
                         constant TensorShape& shape [[buffer(3)]],
                         ushort grid [[thread_position_in_grid]]) {
    int len = shape.dim0;
    
    int index = grid << 2;
    
    if (index >= len) {
        return;
    }
    
    half a = ab.a;
    half b = ab.b;
    
    device half4 *inputV  = (device half4*)(input  + index);
    device half4 *outputV = (device half4*)(output + index);
    
    half4 bV = {b, b, b, b};
    
    outputV[0] = a * inputV[0] + bV;
}

kernel void brouLinear2D(device half *input          [[buffer(0)]],
                         device half *output         [[buffer(1)]],
                         constant AB& ab             [[buffer(2)]],
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

    half a = ab.a;
    half b = ab.b;
    
    half4 bV = {b, b, b, b};
    
    int maxJ = min(y + 4, height);
    
    for (int j = y; j < maxJ; ++j) {
        device half4 *inputV  = (device half4*)(input  + j * width + x);
        device half4 *outputV = (device half4*)(output + j * width + x);
        
        outputV[0] = a * inputV[0] + bV;
    }
}

kernel void brouLinear3D(device half *input          [[buffer(0)]],
                         device half *output         [[buffer(1)]],
                         constant AB& ab             [[buffer(2)]],
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
    
    half a = ab.a;
    half b = ab.b;
    
    half4 bV = b;
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            int offset = (j * width + i) * channel + z;
            
            device half4 *inputV  = (device half4*)(input  + offset);
            device half4 *outputV = (device half4*)(output + offset);

            outputV[0] = a * inputV[0] + bV;
        }
    }
}



















