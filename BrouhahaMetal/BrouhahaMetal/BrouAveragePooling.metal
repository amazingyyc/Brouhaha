#if defined(real) && defined(real4) && defined(BROU)

/**
 * every thread deal with 4 output
 * the input dimesnion is (inputHeight, inputWidth, channel)
 * the ouput dimesnion is (outputHeight, outputWidth, channel)
 */
kernel void BROU(AveragePooling)(device real *input                          [[buffer(0)]],
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
    
    int inputLeft = x * convolutionShape.strideX - convolutionShape.padLeft;
    int inputTop  = y * convolutionShape.strideY - convolutionShape.padTop;
    
    int kernelHeight = convolutionShape.kernelHeight;
    int kernelWidth  = convolutionShape.kernelWidth;
    
    int inputRight  = inputLeft + kernelWidth;
    int inputBottom = inputTop  + kernelHeight;
    
    inputTop  = max(0, inputTop);
    inputLeft = max(0, inputLeft);
    
    inputBottom = min(inputHeight, inputBottom);
    inputRight  = min(inputWidth, inputRight);
    
    real4 sum = 0;
    
    for (int inY = inputTop; inY < inputBottom; ++inY) {
        for (int inX = inputLeft; inX < inputRight; ++inX) {
            device real4 *inputV = (device real4*)(input + (inY * inputWidth + inX) * inputChannel + z);
            
            sum += inputV[0];
        }
    }
    
    device real4 *outputV = (device real4*)(output + (y * outputWidth + x) * outputChannel + z);
    
    outputV[0] = static_cast<real4>(sum / (1.0 * kernelHeight * kernelWidth));
}

kernel void BROU(AveragePoolingWithoutPad)(device real *input                          [[buffer(0)]],
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
    
    int inputLeft = x * convolutionShape.strideX - convolutionShape.padLeft;
    int inputTop  = y * convolutionShape.strideY - convolutionShape.padTop;
    
    int kernelHeight = convolutionShape.kernelHeight;
    int kernelWidth  = convolutionShape.kernelWidth;
    
    int inputRight  = inputLeft + kernelWidth;
    int inputBottom = inputTop  + kernelHeight;
    
    inputTop  = max(0, inputTop);
    inputLeft = max(0, inputLeft);
    
    inputBottom = min(inputHeight, inputBottom);
    inputRight  = min(inputWidth, inputRight);
    
    real4 sum = 0;
    
    for (int inY = inputTop; inY < inputBottom; ++inY) {
        for (int inX = inputLeft; inX < inputRight; ++inX) {
            device real4 *inputV = (device real4*)(input + (inY * inputWidth + inX) * inputChannel + z);
            
            sum += inputV[0];
        }
    }
    
    device real4 *outputV = (device real4*)(output + (y * outputWidth + x) * outputChannel + z);
    
    outputV[0] = static_cast<real4>(sum / (1.0 * (inputBottom - inputTop) * (inputRight - inputLeft)));
}

#endif












