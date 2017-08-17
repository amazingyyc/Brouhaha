/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the PReLu operate
 */
#include <metal_stdlib>

#include "BrouStruct.metal"

using namespace metal;

typedef struct {
    half a;
} A;

/**
 * for 1d shape every thread output 1X4
 */
kernel void brouPReLu1D(device half *input           [[buffer(0)]],
                        device half *output          [[buffer(1)]],
                        constant A& a                [[buffer(2)]],
                        constant TensorShape& shape  [[buffer(3)]],
                        ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;
    
    if (index >= shape.dim0) {
        return;
    }
    
    half realA = a.a;
    
    half4 in = ((device half4*)(input  + index))[0];
    half4 out;
    
    out.x = (in.x > 0) ? (in.x) : (realA * in.x);
    out.y = (in.y > 0) ? (in.y) : (realA * in.y);
    out.z = (in.z > 0) ? (in.z) : (realA * in.z);
    out.w = (in.w > 0) ? (in.w) : (realA * in.w);
    
    device half4 *outputV = (device half4*)(output + index);
    
    outputV[0] = out;
}

/**
 * every thread output 4X4 block
 * the width is timed by 4
 */
kernel void brouPReLu2D(device half *input           [[buffer(0)]],
                        device half *output          [[buffer(1)]],
                        constant A& a                [[buffer(2)]],
                        constant TensorShape& shape  [[buffer(3)]],
                        ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;

    int x = grid.x << 2;
    int y = grid.y << 2;

    if (y >= height || x >= width) {
        return;
    }

    half realA = a.a;
    
    int maxJ = min(y + 4, height);

    for (int j = y; j < maxJ; ++j) {
        int offset = j * width + x;

        half4 in = ((device half4*)(input  + offset))[0];
        half4 out;
        
        out.x = (in.x > 0) ? (in.x) : (realA * in.x);
        out.y = (in.y > 0) ? (in.y) : (realA * in.y);
        out.z = (in.z > 0) ? (in.z) : (realA * in.z);
        out.w = (in.w > 0) ? (in.w) : (realA * in.w);
        
        device half4 *outputV = (device half4*)(output + offset);

        outputV[0] = out;
    }
}

/**
 * every thread output 4X4X4 block
 * the channel must be timed by 4
 */
kernel void brouPReLu3D(device half *input           [[buffer(0)]],
                        device half *output          [[buffer(1)]],
                        constant A& a                [[buffer(2)]],
                        constant TensorShape& shape  [[buffer(3)]],
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

    half realA = a.a;
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);

    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            int offset = (j * width + i) * channel + z;
            
            half4 in = ((device half4*)(input  + offset))[0];
            half4 out;
            
            out.x = (in.x > 0) ? (in.x) : (realA * in.x);
            out.y = (in.y > 0) ? (in.y) : (realA * in.y);
            out.z = (in.z > 0) ? (in.z) : (realA * in.z);
            out.w = (in.w > 0) ? (in.w) : (realA * in.w);
            
            device half4 *outputV = (device half4*)(output + offset);
            
            outputV[0] = out;
        }
    }
}

