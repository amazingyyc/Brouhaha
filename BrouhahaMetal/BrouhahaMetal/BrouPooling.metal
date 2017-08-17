/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the pooling operate
 */
#include <metal_stdlib>

using namespace metal;

/**
 * the input'd dimension is (inputHeight, inputWidth, channel)
 */
constant int inputHeight[[function_constant(0)]];
constant int inputWidth[[function_constant(1)]];

/**
 * the input'd dimension is (outputHeight, outputWidth, channel)
 */
constant int outputHeight[[function_constant(2)]];
constant int outputWidth[[function_constant(3)]];

constant int channel[[function_constant(4)]];

/**
 * the kernel window's size
 */
constant int kernelHeight[[function_constant(5)]];
constant int kernelWidth[[function_constant(6)]];

/**
 * the pad of the input
 */
constant int padLeft[[function_constant(7)]];
constant int padTop[[function_constant(8)]];

/**
 * the step
 */
constant int strideX[[function_constant(9)]];
constant int strideY[[function_constant(10)]];

/**
 * channelX4 >= channel and timed by 4
 */
constant int channelX4[[function_constant(11)]];

/**
 * every thread deal with 4 output
 * the input dimesnion is (inputHeight, inputWidth, channelX4)
 * the ouput dimesnion is (outputHeight, outputWidth, channelX4)
 */
kernel void brouMaxPooling(device half *input    [[buffer(0)]],
                           device half *output   [[buffer(1)]],
                           ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x;
    int y = grid.y;
    int z = grid.z << 2;

    if (x >= outputWidth || y >= outputHeight || z >= channel) {
        return;
    }

    int inputLeft = x * strideX - padLeft;
    int inputTop  = y * strideY - padTop;

    int inputRight  = inputLeft + kernelWidth;
    int inputBottom = inputTop  + kernelHeight;

    inputLeft = max(0, inputLeft);
    inputTop  = max(0, inputTop);

    inputRight  = min(inputWidth, inputRight);
    inputBottom = min(inputHeight, inputBottom);

    int inputAdd = inputWidth * channelX4;
    device half4 *inputV = (device half4*)(input + (inputTop * inputWidth + inputLeft) * channelX4 + z);

    half4 out = inputV[0];

    for (int i = inputTop; i < inputBottom; ++i) {
        device half4 *inputOffsetV = inputV;

        for (int j = inputLeft; j < inputRight; ++j) {
            out = max(out, inputOffsetV[0]);

            inputOffsetV = (device half4*)((device half*)inputOffsetV + channelX4);
        }

        inputV = (device half4*)((device half*)inputV + inputAdd);
    }

    device half4 *outputV = (device half4*)(output + (y * outputWidth + x) * channelX4 + z);
    outputV[0] = out;
}

/**
 * every thread deal with 4 output
 * the input dimesnion is (inputHeight, inputWidth, channelX4)
 * the ouput dimesnion is (outputHeight, outputWidth, channelX4)
 */
kernel void brouAveragePooling(device half *input    [[buffer(0)]],
                               device half *output   [[buffer(1)]],
                               ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x;
    int y = grid.y;
    int z = grid.z << 2;

    if (x >= outputWidth || y >= outputHeight || z >= channel) {
        return;
    }

    int inputLeft = x * strideX - padLeft;
    int inputTop  = y * strideY - padTop;

    int inputRight  = inputLeft + kernelWidth;
    int inputBottom = inputTop  + kernelHeight;

    inputLeft = max(0, inputLeft);
    inputTop  = max(0, inputTop);

    inputRight  = min(inputWidth, inputRight);
    inputBottom = min(inputHeight, inputBottom);

    int inputAdd = inputWidth * channelX4;
    device half4 *inputV = (device half4*)(input + (inputTop * inputWidth + inputLeft) * channelX4 + z);

    half4 sum = 0;

    for (int i = inputTop; i < inputBottom; ++i) {
        device half4 *inputOffsetV = inputV;

        for (int j = inputLeft; j < inputRight; ++j) {
            sum += inputOffsetV[0];

            inputOffsetV = (device half4*)((device half*)inputOffsetV + channelX4);
        }

        inputV = (device half4*)((device half*)inputV + inputAdd);
    }

    device half4 *outputV = (device half4*)(output + (y * outputWidth + x) * channelX4 + z);
    outputV[0] = static_cast<half4>(sum / (1.0 * kernelHeight * kernelWidth));
}

/**
 * every thread deal with 4 output
 * the input dimesnion is (inputHeight, inputWidth, channelX4)
 * the ouput dimesnion is (outputHeight, outputWidth, channelX4)
 */
kernel void brouAveragePoolingWithoutPad(device half *input    [[buffer(0)]],
                                         device half *output   [[buffer(1)]],
                                         ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x;
    int y = grid.y;
    int z = grid.z << 2;

    if (x >= outputWidth || y >= outputHeight || z >= channel) {
        return;
    }

    int inputLeft = x * strideX - padLeft;
    int inputTop  = y * strideY - padTop;

    int inputRight  = inputLeft + kernelWidth;
    int inputBottom = inputTop  + kernelHeight;

    inputLeft = max(0, inputLeft);
    inputTop  = max(0, inputTop);

    inputRight  = min(inputWidth, inputRight);
    inputBottom = min(inputHeight, inputBottom);

    int inputAdd = inputWidth * channelX4;
    device half4 *inputV = (device half4*)(input + (inputTop * inputWidth + inputLeft) * channelX4 + z);

    half4 sum = 0;

    for (int i = inputTop; i < inputBottom; ++i) {
        device half4 *inputOffsetV = inputV;

        for (int j = inputLeft; j < inputRight; ++j) {
            sum += inputOffsetV[0];

            inputOffsetV = (device half4*)((device half*)inputOffsetV + channelX4);
        }

        inputV = (device half4*)((device half*)inputV + inputAdd);
    }

    device half4 *outputV = (device half4*)(output + (y * outputWidth + x) * channelX4 + z);
    outputV[0] = static_cast<half4>(sum / (1.0 * (inputBottom - inputTop) * (inputRight - inputLeft)));
}






