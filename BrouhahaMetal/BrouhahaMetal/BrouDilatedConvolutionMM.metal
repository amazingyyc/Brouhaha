#if defined(real) && defined(real4) && defined(BROU)

/**
 * the input's dimension is (inputHeight, intputWidth, intputChannelX4)
 * it will be convert to matrix that is (kernelHeight * kernelWidth * inputChannelX4, [outputHeight * outputWidth]X4)
 * the matrix will be col-major
 *
 * the convolutoin is not equal the real convolution in math
 * like input is (a, b, c) the kernel is (i, j, k)
 * the convolution in math is output = a*k + b*j + c*i
 * but in brouhaha the convolution will be output = a*i + b*j + c*k
 */
inline real4 BROU(GetDilatedConvolutionVector4FromInput)(device real *data, int height, int width, int channel, int y, int x, int z) {
    if (0 > y || 0 > x || 0 > z || y >= height || x >= width || z >= channel) {
        return 0;
    }
    
    device real4 *dataV = (device real4*)(data + (y * width + x) * channel + z);
    
    return dataV[0];
}

kernel void BROU(DilatedConvolutionInput2Matrix)(device real *input                            [[buffer(0)]],
                                                 device real *matrix                           [[buffer(1)]],
                                                 constant TensorShape& inputShape              [[buffer(2)]],
                                                 constant TensorShape& outputShape             [[buffer(3)]],
                                                 constant ConvolutionShape& convolutionShape   [[buffer(4)]],
                                                 ushort grid [[thread_position_in_grid]]) {
    int outputHeight = outputShape.dim0;
    int outputWidth  = outputShape.dim1;
    
    /**every thread handle 4 col output*/
    int col = grid << 2;
    
    if (col >= outputHeight * outputWidth) {
        return;
    }
    
    int inputHeight  = inputShape.dim0;
    int inputWidth   = inputShape.dim1;
    int inputChannel = inputShape.dim2;
    
    int padTop  = convolutionShape.padTop;
    int padLeft = convolutionShape.padLeft;
    
    int strideY = convolutionShape.strideY;
    int strideX = convolutionShape.strideX;
    
    int kernelHeight = convolutionShape.kernelHeight;
    int kernelWidth  = convolutionShape.kernelWidth;
    
    int dilatedX = convolutionShape.dilatedX;
    int dilatedY = convolutionShape.dilatedY;
    
    int inputX0 = (col       % outputWidth) * strideX - padLeft;
    int inputY0 = (col       / outputWidth) * strideY - padTop;
    int inputX1 = ((col + 1) % outputWidth) * strideX - padLeft;
    int inputY1 = ((col + 1) / outputWidth) * strideY - padTop;
    int inputX2 = ((col + 2) % outputWidth) * strideX - padLeft;
    int inputY2 = ((col + 2) / outputWidth) * strideY - padTop;
    int inputX3 = ((col + 3) % outputWidth) * strideX - padLeft;
    int inputY3 = ((col + 3) / outputWidth) * strideY - padTop;
    
    int matrixCol = (outputHeight * outputWidth + 3) / 4 * 4;
    device real4 *matrixV = (device real4*)(matrix + col);
    
    real4 inputV0, inputV1, inputV2, inputV3;
    
    for (int y = 0; y < kernelHeight; ++y) {
        for (int x = 0; x < kernelWidth; ++x) {
            for (int c = 0; c < inputChannel; c += 4) {
                inputV0 = BROU(GetDilatedConvolutionVector4FromInput)(input,inputHeight,inputWidth,inputChannel,inputY0+y*dilatedY,inputX0+x*dilatedX,c);
                inputV1 = BROU(GetDilatedConvolutionVector4FromInput)(input,inputHeight,inputWidth,inputChannel,inputY1+y*dilatedY,inputX1+x*dilatedX,c);
                inputV2 = BROU(GetDilatedConvolutionVector4FromInput)(input,inputHeight,inputWidth,inputChannel,inputY2+y*dilatedY,inputX2+x*dilatedX,c);
                inputV3 = BROU(GetDilatedConvolutionVector4FromInput)(input,inputHeight,inputWidth,inputChannel,inputY3+y*dilatedY,inputX3+x*dilatedX,c);
                
                matrixV[0] = {inputV0.x, inputV1.x, inputV2.x, inputV3.x};
                
                matrixV = (device real4*)((device real*)matrixV + matrixCol);
                matrixV[0] = {inputV0.y, inputV1.y, inputV2.y, inputV3.y};
                
                matrixV = (device real4*)((device real*)matrixV + matrixCol);
                matrixV[0] = {inputV0.z, inputV1.z, inputV2.z, inputV3.z};
                
                matrixV = (device real4*)((device real*)matrixV + matrixCol);
                matrixV[0] = {inputV0.w, inputV1.w, inputV2.w, inputV3.w};
            }
        }
    }
}

#endif















