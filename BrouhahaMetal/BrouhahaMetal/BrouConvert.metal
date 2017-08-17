/**
 * BrouConvert.metal
 * BrouhahaMetal
 *
 * Created by yanyuanchi on 2017/6/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * convert the float32 to float16 or float16=>float32
 */

#include <metal_stdlib>
using namespace metal;

/**
 * if the input and the output is 3d
 * then the input ant output's dimension is (height, width, channelX4)
 *
 * if the input and output is 1d than the dimension is (channelX4, 1)
 */
constant int height  [[function_constant(0)]];
constant int width   [[function_constant(1)]];
constant int channel [[function_constant(2)]];

constant int channelX4 [[function_constant(3)]];

kernel void convertFloatToHalf3D(device float *input    [[buffer(0)]],
                                 device half  *output   [[buffer(1)]],
                                 ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x;
    int y = grid.y;
    int z = grid.z << 2;

    if (x >= width || y >= height || z >= channel) {
        return;
    }

    device float4 *inputV = (device float4*)(input  + (y * width + x) * channelX4 + z);
    device half4 *outputV = (device  half4*)(output + (y * width + x) * channelX4 + z);

    outputV[0] = static_cast<half4>(inputV[0]);
}

kernel void convertFloatToHalf1D(device float *input    [[buffer(0)]],
                                 device half  *output   [[buffer(1)]],
                                 ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;

    if (index >= channel) {
        return;
    }

    device float4 *inputV  = (device float4*)(input  + index);
    device half4  *outputV = (device  half4*)(output + index);

    outputV[0] = static_cast<half4>(inputV[0]);
}

kernel void convertHalfToFloat3D(device half  *input[[buffer(0)]],
                                 device float *output[[buffer(1)]],
                                 ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x;
    int y = grid.y;
    int z = grid.z << 2;

    if (x >= width || y >= height || z >= channel) {
        return;
    }

    device half4  *inputV  = (device  half4*)(input  + (y * width + x) * channelX4 + z);
    device float4 *outputV = (device float4*)(output + (y * width + x) * channelX4 + z);

    outputV[0] = static_cast<float4>(inputV[0]);
}

kernel void convertHalfToFloat1D(device half  *input[[buffer(0)]],
                                 device float *output[[buffer(1)]],
                                 ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;

    if (index >= channel) {
        return;
    }

    device half4 *inputV   = (device  half4*)(input  + index);
    device float4 *outputV = (device float4*)(output + index);

    outputV[0] = static_cast<float4>(inputV[0]);
}












