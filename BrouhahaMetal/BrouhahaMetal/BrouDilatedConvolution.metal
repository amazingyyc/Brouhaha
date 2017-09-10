/**
 * handle the dialted convolution
 */

#if defined(real) && defined(real4) && defined(BROU)

kernel void BROU(DilatedConvolution)(device real *input                           [[buffer(0)]],
                                     device real *kerne                           [[buffer(1)]],
                                     device real *bia                             [[buffer(2)]],
                                     device real *output                          [[buffer(3)]],
                                     constant TensorShape& inputShape             [[buffer(4)]],
                                     constant TensorShape& outputShape            [[buffer(5)]],
                                     constant ConvolutionShape& convolutionShape  [[buffer(6)]],
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
    
    int strideY = convolutionShape.strideY;
    int strideX = convolutionShape.strideX;
    
    int dilatedY = convolutionShape.dilatedY;
    int dilatedX = convolutionShape.dilatedX;
    
    int maxOutY = min(y + 4, outputHeight);
    int maxOutX = min(x + 4, outputWidth);
    
    real4 biasV = convolutionShape.haveBias ? ((device real4*)(bia + z))[0] : 0;
    
    for (int outY = y; outY < maxOutY; ++outY) {
        for (int outX = x; outX < maxOutX; ++outX) {
            real4 out = biasV;
            
            int inputTop =  -padTop  + outY * strideY;
            int inputLeft = -padLeft + outX * strideX;
            
            int inputBottom = inputTop  + dilatedY * (kernelHeight - 1);
            int inputRight  = inputLeft + dilatedX * (kernelWidth  - 1);
            
            int kernelTop = (inputTop  >= 0) ? 0 : -inputTop;
            int kenelLeft = (inputLeft >= 0) ? 0 : -inputLeft;
            
            inputTop  = max(0, inputTop);
            inputLeft = max(0, inputLeft);
            
            inputBottom = min(inputHeight - 1, inputBottom);
            inputRight  = min(inputWidth  - 1, inputRight);
            
            for (int inY = inputTop, kernelY = kernelTop; inY <= inputBottom; inY += dilatedY, ++kernelY) {
                for (int inX = inputLeft, kernelX = kenelLeft; inX <= inputRight; inX += dilatedX, ++kernelX) {
                    device real *inputOffset = input + (inY * inputWidth + inX) * inputChannel;
                    
                    device real *kernelOffset0 = kerne + (((z  ) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    device real *kernelOffset1 = kerne + (((z+1) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    device real *kernelOffset2 = kerne + (((z+2) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    device real *kernelOffset3 = kerne + (((z+3) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannel;
                    
                    for (int c = 0; c < inputChannel; c += 4) {
                        real4 inV = ((device real4*)(inputOffset))[0];
                        
                        real4 kernelV0 = ((device real4*)(kernelOffset0))[0];
                        real4 kernelV1 = ((device real4*)(kernelOffset1))[0];
                        real4 kernelV2 = ((device real4*)(kernelOffset2))[0];
                        real4 kernelV3 = ((device real4*)(kernelOffset3))[0];
                        
                        out.x += dot(inV, kernelV0);
                        out.y += dot(inV, kernelV1);
                        out.z += dot(inV, kernelV2);
                        out.w += dot(inV, kernelV3);
                        
                        inputOffset   += 4;
                        
                        kernelOffset0 += 4;
                        kernelOffset1 += 4;
                        kernelOffset2 += 4;
                        kernelOffset3 += 4;
                    }
                }
            }
            
            device real4 *outputV = (device real4*)(output + (outY * outputWidth + outX) * outputChannel + z);
            
            outputV[0] = out;
        }
    }
}

#endif











