/**
 * Created by yanyuanchi on 2017/7/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the batch Normalization layer
 */

#include <metal_stdlib>
using namespace metal;

/**
 * in testing the mean and variance will be knowed
 * if don't know the training mean and variance, the mean and variance will be calculate by
 * brouCalculateMeanAndVariance3D
 *
 * the alpha and beta is knowed
 *
 * output = alpha * (input - mean) / (sqrt(variance + epsilon)) + beta
 */

/**
 * if the input and the output is 3d
 * then the input ant output's dimension is (height, width, channelX4)
 *
 * if the input and output is 1d than the dimension is (channelX4, 1)
 *
 * the mean and variance's dimension is (channelX4, 1)
 */
constant int height  [[function_constant(0)]];
constant int width   [[function_constant(1)]];
constant int channel [[function_constant(2)]];

constant int channelX4 [[function_constant(3)]];

constant float epsilon [[function_constant(4)]];

/**
 * calculate the input mean and "varicance"
 */
kernel void brouCalculateMeanAndVariance3D(device half *input    [[buffer(0)]],
                                           device half *mean     [[buffer(1)]],
                                           device half *variance [[buffer(2)]],
                                           ushort grid [[thread_position_in_grid]]) {
    int z = grid << 2;
    
    if (z >= channel) {
        return;
    }
    
    /**use float to store sum of input*/
    float4 sum = 0;
    
    /**calcualte mean*/
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            half4 inputV = ((device half4*)(input + (y * width + x) * channelX4 + z))[0];
            
            sum += (static_cast<float4>(inputV));
        }
    }
    
    half4 meanV = static_cast<half4>(sum / (1.0 * height * width));
    
    sum = 0;
    
    /**calculate variance*/
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            half4 inputV = ((device half4*)(input + (y * width + x) * channelX4 + z))[0];
            half4 differ = inputV - meanV;
            
            sum += static_cast<float4>(differ * differ);
        }
    }
    
    half4 varianceV = static_cast<half4>(1.0 / sqrt(sum / (1.0 * height * width) + epsilon));
    
    device half4 *mean4     = (device half4*)(mean     + z);
    device half4 *variance4 = (device half4*)(variance + z);
    
    mean4[0]     = meanV;
    variance4[0] = varianceV;
}

/**
 * every thread handle 4X4X4 block
 */
kernel void brouBatchNormalization3D(device half *input    [[buffer(0)]],
                                     device half *output   [[buffer(1)]],
                                     device half *mean     [[buffer(2)]],
                                     device half *variance [[buffer(3)]],
                                     device half *alpha    [[buffer(4)]],
                                     device half *beta     [[buffer(5)]],
                                     ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= width || y >= height || z >= channel) {
        return;
    }
    
    half4 meanV     = ((device half4*)(mean     + z))[0];
    half4 varianceV = ((device half4*)(variance + z))[0];
    half4 alphaV    = ((device half4*)(alpha    + z))[0];
    half4 betaV     = ((device half4*)(beta     + z))[0];
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            device half4 *inputV  = (device half4*)(input  + (j * width + i) * channelX4 + z);
            device half4 *outputV = (device half4*)(output + (j * width + i) * channelX4 + z);
            
            outputV[0] = alphaV * (inputV[0] - meanV) * varianceV + betaV;
        }
    }
}












