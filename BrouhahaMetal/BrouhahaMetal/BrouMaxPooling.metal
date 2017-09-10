/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the max pooling operate
 */
#if defined(real) && defined(real4) && defined(BROU)

/**
 * every thread deal with 4 output
 * the input dimesnion is (inputHeight, inputWidth, channelX4)
 * the ouput dimesnion is (outputHeight, outputWidth, channelX4)
 */
kernel void BROU(MaxPooling)(device real *input                          [[buffer(0)]],
                             device real *output                         [[buffer(1)]],
                             constant TensorShape& inputShape            [[buffer(2)]],
                             constant TensorShape& outputShape           [[buffer(3)]],
                             constant ConvolutionShape& convolutionShape [[buffer(4)]],
                             ushort3 grid [[thread_position_in_grid]]) {
    int outputHeight  = outputShape.dim0;
    int outputWidth   = outputShape.dim1;
    int outputChannel = outputShape.dim2;
    
    int x = grid.x;
    int y = grid.y;
    int z = grid.z << 2;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
 
    int inputHeight  = inputShape.dim0;
    int inputWidth   = inputShape.dim1;
    int inputChannel = inputShape.dim2;
    
    int inputTop  = -convolutionShape.padTop  + convolutionShape.strideY * y;
    int inputLeft = -convolutionShape.padLeft + convolutionShape.strideX * x;
    
    int inputBottom = inputTop  + convolutionShape.kernelHeight;
    int inputRight  = inputLeft + convolutionShape.kernelWidth;
    
    inputTop  = max(0, inputTop);
    inputLeft = max(0, inputLeft);
    
    inputBottom = min(inputHeight, inputBottom);
    inputRight  = min(inputWidth, inputRight);
    
    device real4 *inputV = (device real4*)(input + (inputTop * inputWidth + inputLeft) * inputChannel + z);
    
    real4 out = inputV[0];
    
    for (int inY = inputTop; inY < inputBottom; ++inY) {
        for (int inX = inputLeft; inX < inputRight; ++inX) {
            inputV = (device real4*)(input + (inY * inputWidth + inX) * inputChannel + z);
            
            out = max(out, inputV[0]);
        }
    }
    
    device real4 *outputV = (device real4*)(output + (y * outputWidth + x) * outputChannel + z);
    
    outputV[0] = out;
}

#endif







