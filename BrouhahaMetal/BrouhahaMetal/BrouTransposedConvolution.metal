/**
 * BrouTransposedConvolution.metal
 * BrouhahaMetal
 *
 * Created by yanyuanchi on 2017/7/30.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#include <metal_stdlib>

#include "BrouStruct.metal"

using namespace metal;

/**
 * ref:A guide to convolution arithmetic for deep learning
 *
 * a convolution that with pad, stride, kernel, input, output, than it will hava a corresponding transposed convolution
 * that kernel on the output ((stride - 1) zeros are inserted between output units)
 * and pad' = kernel - 1 - pad, stride' = 1, kernel' = kernel
 */

/**
 * for the input, output, kernel's diemension, the innermost dimension must timed by 4
 */
kernel void brouTransposedConvolution(device half *input   [[buffer(0)]],
                                      device half *kerne   [[buffer(1)]],
                                      device half *bia     [[buffer(2)]],
                                      device half *output  [[buffer(3)]],
                                      constant TensorShape& inputShape  [[buffer(4)]],
                                      constant TensorShape& outputShape [[buffer(5)]],
                                      constant ConvolutionShape& convolutionShape [[buffer(6)]],
                                      ushort3 grid [[thread_position_in_grid]]) {
    int outputHeight  = outputShape.dim0;
    int outputWidth   = outputShape.dim1;
    int outputChannel = outputShape.dim2;

    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;

    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }

    int inputHeight  = inputShape.dim0;
    int inputWidth   = inputShape.dim1;
    int inputChannel = inputShape.dim2;

    int kernelHeight = convolutionShape.kernelHeight;
    int kernelWidth  = convolutionShape.kernelWidth;

    int padTop  = convolutionShape.padTop;
    int padLeft = convolutionShape.padLeft;

    int insertY = convolutionShape.insertY;
    int insertX = convolutionShape.insertX;

    int insertYAdd1 = insertY + 1;
    int insertXAdd1 = insertX + 1;

    int fakeInputHeight = inputHeight + (inputHeight - 1) * insertY;
    int fakeInputWidth  = inputWidth  + (inputWidth  - 1) * insertX;

    int maxOutY = min(y + 4, outputHeight);
    int maxOutX = min(x + 4, outputWidth);

    /**if have a bias*/
    half4 biasV = convolutionShape.haveBias ? ((device half4*)(bia + z))[0] : 0;

    for (int outY = y; outY < maxOutY; ++outY) {
        for (int outX = x; outX < maxOutX; ++outX) {
            /**store the out*/
            half4 out = 0;

            int inputTop  = -padTop  + outY;
            int inputLeft = -padLeft + outX;

            int inputBottom = min(inputTop  + kernelHeight, fakeInputHeight);
            int inputRight  = min(inputLeft + kernelWidth,  fakeInputWidth);

            int realInputTop  = (0 > inputTop)  ? 0 : ((inputTop  + insertY) / insertYAdd1 * insertYAdd1);
            int realInputLeft = (0 > inputLeft) ? 0 : ((inputLeft + insertX) / insertXAdd1 * insertXAdd1);

            int kernelTop  = realInputTop  - inputTop;
            int kernelLeft = realInputLeft - inputLeft;

            for (int inY = realInputTop, kernelY = kernelTop; inY < inputBottom; inY += insertYAdd1, kernelY += insertYAdd1) {
                for (int inX = realInputLeft, kernelX = kernelLeft; inX < inputRight; inX += insertXAdd1, kernelX += insertXAdd1) {
                    int realInY = inY / insertYAdd1;
                    int realInX = inX / insertXAdd1;

                    device half *inOffset = input + (realInY * inputWidth + realInX) * inputChannel;
                    
                    device half *kernelOffset0 = kerne + ((z       * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    device half *kernelOffset1 = kerne + (((z + 1) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    device half *kernelOffset2 = kerne + (((z + 2) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    device half *kernelOffset3 = kerne + (((z + 3) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    
                    for (int c = 0; c < inputChannel; c += 4) {
                        half4 inV = ((device half4*)(inOffset))[0];
                        
                        half4 kernelV0 = ((device half4*)(kernelOffset0))[0];
                        half4 kernelV1 = ((device half4*)(kernelOffset1))[0];
                        half4 kernelV2 = ((device half4*)(kernelOffset2))[0];
                        half4 kernelV3 = ((device half4*)(kernelOffset3))[0];

                        out.x += dot(inV, kernelV0);
                        out.y += dot(inV, kernelV1);
                        out.z += dot(inV, kernelV2);
                        out.w += dot(inV, kernelV3);
                    
                        inOffset += 4;
                        
                        kernelOffset0 += 4;
                        kernelOffset1 += 4;
                        kernelOffset2 += 4;
                        kernelOffset3 += 4;
                    }
                }
            }

            device half4 *outputV = (device half4*)(output + (outY * outputWidth + outX) * outputChannel + z);

            outputV[0] = out + biasV;
        }
    }
}






















