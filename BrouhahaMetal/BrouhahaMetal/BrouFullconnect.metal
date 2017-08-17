/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the fullconnet operate
 */

#include <metal_stdlib>

using namespace metal;

/**the input's dimension (inputChannel, 1)*/
constant int inputChannel[[function_constant(0)]];

/**the output's dimension (outputChannel, 1)*/
constant int outputChannel[[function_constant(1)]];

constant int inputChannelX4[[function_constant(2)]];
constant int outputChannelX4[[function_constant(3)]];

/**
 * every thread will deal with 4 output
 * the input's dimesnion is (inputChannelX4, 1)
 * the output's dimension is (outputChannelX4, 1)
 * the weigths's dimesnion is (outputChannelX4, inputChannelX4)
 * the bias's dimension is (outputChannelX4, 1)
 */
kernel void brouFullconnectBlock(device half *input      [[buffer(0)]],
                                 device half *weights    [[buffer(1)]],
                                 device half *bia        [[buffer(2)]],
                                 device half *output     [[buffer(3)]],
                                 ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;

    if (index >= outputChannel) {
        return;
    }

    device half4 *inputV   = (device half4*)input;
    device half4 *weightsV = (device half4*)(weights + inputChannelX4 * index);

    half4 out = 0;
    half4 in, wei0, wei1, wei2, wei3;

    int loopCount = (inputChannel + 3) / 4;

    do {
        in = inputV[0];

        device half4 *weightsVOffset = weightsV;

        wei0 = weightsVOffset[0]; weightsVOffset = (device half4*)((device half*)weightsVOffset + inputChannelX4);
        wei1 = weightsVOffset[0]; weightsVOffset = (device half4*)((device half*)weightsVOffset + inputChannelX4);
        wei2 = weightsVOffset[0]; weightsVOffset = (device half4*)((device half*)weightsVOffset + inputChannelX4);
        wei3 = weightsVOffset[0];

        out.x += dot(in, wei0);
        out.y += dot(in, wei1);
        out.z += dot(in, wei2);
        out.w += dot(in, wei3);

        inputV++;
        weightsV++;
    } while(--loopCount);

    device half4 *outputV = (device half4*)(output + index);
    device half4 *biaV    = (device half4*)(bia + index);

    outputV[0] = out + biaV[0];
}


kernel void brouFullconnectBlockWithoutBias(device half *input      [[buffer(0)]],
                                            device half *weights    [[buffer(1)]],
                                            device half *output     [[buffer(2)]],
                                            ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;

    if (index >= outputChannel) {
        return;
    }

    device half4 *inputV   = (device half4*)input;
    device half4 *weightsV = (device half4*)(weights + inputChannelX4 * index);

    half4 out = 0;
    half4 in, wei0, wei1, wei2, wei3;

    int loopCount = (inputChannel + 3) / 4;

    do {
        in = inputV[0];

        device half4 *weightsVOffset = weightsV;

        wei0 = weightsVOffset[0]; weightsVOffset = (device half4*)((device half*)weightsVOffset + inputChannelX4);
        wei1 = weightsVOffset[0]; weightsVOffset = (device half4*)((device half*)weightsVOffset + inputChannelX4);
        wei2 = weightsVOffset[0]; weightsVOffset = (device half4*)((device half*)weightsVOffset + inputChannelX4);
        wei3 = weightsVOffset[0];

        out.x += dot(in, wei0);
        out.y += dot(in, wei1);
        out.z += dot(in, wei2);
        out.w += dot(in, wei3);

        inputV++;
        weightsV++;
    } while(--loopCount);

    device half4 *outputV = (device half4*)(output + index);

    outputV[0] = out;
}

/***************************************************************************************/
/**for test*/
/***************************************************************************************/

/**
 * the weight's diemension is (outputChannel, inputChannel)
 * the weight is row-major matrix
 * the bias's dimension is (outputChannle, 1)
 */

kernel void brouFullconnect(device half *input      [[buffer(0)]],
                            device half *weights    [[buffer(1)]],
                            device half *bia        [[buffer(2)]],
                            device half *output     [[buffer(3)]],
                            uint grid [[thread_position_in_grid]]) {
    if (grid >= outputChannel) {
        return;
    }

    device half *inputData   = input;
    device half *weightsData = weights + grid * inputChannel;

    half sum0 = 0;
    half sum1 = 0;
    half sum2 = 0;
    half sum3 = 0;

    int limit = inputChannel - 3;

    int i = 0;
    for (; i < limit; i += 4) {
        sum0 += inputData[i] * weightsData[i];
        sum1 += inputData[i + 1] * weightsData[i + 1];
        sum2 += inputData[i + 2] * weightsData[i + 2];
        sum3 += inputData[i + 3] * weightsData[i + 3];
    }

    for (; i < inputChannel; ++i) {
        sum0 += inputData[i] * weightsData[i];
    }

    output[grid] = sum0 + sum1 + sum2 + sum3 + bia[grid];
}

/**
 * the weight's diemension is (outputChannel, inputChannel)
 * the weight is row-major matrix
 * the bias's dimension is (outputChannle, 1)
 */

kernel void brouFullconnectWithoutBias(device half *input      [[buffer(0)]],
                                       device half *weights    [[buffer(1)]],
                                       device half *output     [[buffer(2)]],
                                       uint grid [[thread_position_in_grid]]) {
    if (grid >= outputChannel) {
        return;
    }

    device half *inputData   = input;
    device half *weightsData = weights + grid * inputChannel;

    half sum0 = 0;
    half sum1 = 0;
    half sum2 = 0;
    half sum3 = 0;

    int limit = inputChannel - 3;

    int i = 0;
    for (; i < limit; i += 4) {
        sum0 += inputData[i] * weightsData[i];
        sum1 += inputData[i + 1] * weightsData[i + 1];
        sum2 += inputData[i + 2] * weightsData[i + 2];
        sum3 += inputData[i + 3] * weightsData[i + 3];
    }

    for (; i < inputChannel; ++i) {
        sum0 += inputData[i] * weightsData[i];
    }

    output[grid] = sum0 + sum1 + sum2 + sum3;
}











