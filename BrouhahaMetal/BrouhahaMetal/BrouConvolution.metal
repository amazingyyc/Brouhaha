/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the convolution opetator
 */

#if defined(real) && defined(real4) && defined(BROU)

/**
 * every thread deal with 4 X 4 X 4 output
 * the output is spilted to (4, 4, 4) blocks, and every thread handle one block
 * the input's dimension is (inputHeight, inputWidth, intputChannelX4)
 * the output's dimeansion is (outputHeight, outputWidth, outputChannelX4)
 * the weights's dimension is (outputChannelX4, kernelheight, kernelwidth, inputChannelX4)
 * the bias's dimesion is (outputchannleX4, 1), if have
 */

kernel void BROU(Convolution)(device real *input   [[buffer(0)]],
                              device real *kerne   [[buffer(1)]],
                              device real *bia     [[buffer(2)]],
                              device real *output  [[buffer(3)]],
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
    
    int padLeft = convolutionShape.padLeft;
    int padTop  = convolutionShape.padTop;
    
    int strideX = convolutionShape.strideX;
    int strideY = convolutionShape.strideY;
    
    int maxOutY = min(y + 4, outputHeight);
    int maxOutX = min(x + 4, outputWidth);
    
    real4 biasV = convolutionShape.haveBias ? ((device real4*)(bia + z))[0] : 0;
    
    for (int outY = y; outY < maxOutY; ++outY) {
        for (int outX = x; outX < maxOutX; ++outX) {
            real4 out = biasV;
            
            int inputLeft = outX * strideX - padLeft;
            int inputTop  = outY * strideY - padTop;
            
            int inputRight  = inputLeft + kernelWidth;
            int inputBottom = inputTop  + kernelHeight;
            
            int kernelLeft = inputLeft >= 0 ? 0 : -inputLeft;
            int kernelTop  = inputTop  >= 0 ? 0 : -inputTop;
            
            inputLeft = max(0, inputLeft);
            inputTop  = max(0, inputTop);
            
            inputRight  = min(inputWidth, inputRight);
            inputBottom = min(inputHeight, inputBottom);
            
            for (int inY = inputTop, kernelY = kernelTop; inY < inputBottom; ++inY, ++kernelY) {
                for (int inX = inputLeft, kernelX = kernelLeft; inX < inputRight; ++inX, ++kernelX) {
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

/**
 * for special kernel/stride use the special kernel function
 */
inline void BROU(GetVector4FromInput)(device real *input,
                                      int height, int width, int channel,
                                      int y, int x, int z,
                                      thread real4 *vector,
                                      int row, int col) {
    for (int j = 0; j < row; ++j) {
        for (int i = 0; i < col; ++i) {
            int realJ = j + y;
            int realI = i + x;
            
            if (0 > realJ || realJ >= height || 0 > realI || realI > width || z >= channel) {
                vector[j * col + i] = 0;
            } else {
                device real4 *inputV = (device real4*)(input + (realJ * width + realI) * channel + z);
                
                vector[j * col + i] = inputV[0];
            }
        }
    }
}

/**
 * the kernel is 3X3
 */
 inline void BROU(GetKernel3X3Vector)(device real *kerne, int inputChannel, int outputOffset, int channelOffset, real4 vector[3][3]) {
    device real *kernelOffset = kerne + outputOffset * 3 * 3 * inputChannel + channelOffset;

    vector[0][0] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][1] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][2] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[1][0] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][1] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][2] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[2][0] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][1] = ((device real4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][2] = ((device real4*)(kernelOffset))[0];
}

kernel void BROU(ConvolutionKernel3X3)(device real *input   [[buffer(0)]],
                                       device real *kerne   [[buffer(1)]],
                                       device real *bia     [[buffer(2)]],
                                       device real *output  [[buffer(3)]],
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
     
     int padTop  = convolutionShape.padTop;
     int padLeft = convolutionShape.padLeft;
     
     int strideX = convolutionShape.strideX;
     int strideY = convolutionShape.strideY;
     
     real4 in[3][3];
     real4 ker0[3][3],  ker1[3][3], ker2[3][3], ker3[3][3];
     
     real4 biasV = convolutionShape.haveBias ? ((device real4*)(bia + z))[0] : 0;
     
     real4 out[4][4] = {{biasV, biasV, biasV, biasV}, {biasV, biasV, biasV, biasV}, {biasV, biasV, biasV, biasV}, {biasV, biasV, biasV, biasV}};
     
     int maxOutY = min(4, outputHeight - y);
     int maxOutX = min(4, outputWidth  - x);
     
     for (int c = 0; c < inputChannel; c += 4) {
         BROU(GetKernel3X3Vector)(kerne, inputChannel, z    , c, ker0);
         BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 1, c, ker1);
         BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 2, c, ker2);
         BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 3, c, ker3);
         
         for (int outY = 0; outY < maxOutY; ++outY) {
             for (int outX = 0; outX < maxOutX; ++outX) {
                 int inputTop  = -padTop  + strideY * (y + outY);
                 int inputLeft = -padLeft + strideX * (x + outX);
                 
                 BROU(GetVector4FromInput)(input, inputHeight, inputWidth, inputChannel, inputTop, inputLeft, c, (thread real4*)in, 3, 3);
                 
                 out[outY][outX].x += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker0));
                 out[outY][outX].y += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker1));
                 out[outY][outX].z += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker2));
                 out[outY][outX].w += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker3));
             }
         }
     }
     
     for (int outY = 0; outY < maxOutY; ++outY) {
         for (int outX = 0; outX < maxOutX; ++outX) {
             device real4 *outputV = (device real4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannel + z);
             outputV[0] = out[outY][outX];
         }
     }
 }

kernel void BROU(ConvolutionKernel3X3Stride1X1)(device real *input   [[buffer(0)]],
                                                device real *kerne   [[buffer(1)]],
                                                device real *bia     [[buffer(2)]],
                                                device real *output  [[buffer(3)]],
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
    
    int padTop  = convolutionShape.padTop;
    int padLeft = convolutionShape.padLeft;

    real4 in[6][6];
    real4 ker0[3][3],  ker1[3][3], ker2[3][3], ker3[3][3];
    
    real4 biasV = convolutionShape.haveBias ? ((device real4*)(bia + z))[0] : 0;
    
    real4 out[4][4] = {
        {biasV, biasV, biasV, biasV},
        {biasV, biasV, biasV, biasV},
        {biasV, biasV, biasV, biasV},
        {biasV, biasV, biasV, biasV}
    };
    
    int inputTop  = y - padTop;
    int inputLeft = x - padLeft;
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannel; c += 4) {
        BROU(GetVector4FromInput)(input, inputHeight, inputWidth, inputChannel, inputTop, inputLeft, c, (thread real4*)in, 6, 6);
        
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z    , c, ker0);
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 1, c, ker1);
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 2, c, ker2);
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 3, c, ker3);
        
        for (int outY = 0, inY = 0; outY < maxOutY; ++outY, ++inY) {
            for (int outX = 0, inX = 0; outX < maxOutX; ++outX, ++inX) {
                out[outY][outX].x += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker3));
            }
        }
    }
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device real4 *outputV = (device real4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannel + z);
            outputV[0] = out[outY][outX];
        }
    }
}

kernel void BROU(ConvolutionKernel3X3Stride2X2)(device real *input   [[buffer(0)]],
                                                device real *kerne   [[buffer(1)]],
                                                device real *bia     [[buffer(2)]],
                                                device real *output  [[buffer(3)]],
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
    
    int padTop  = convolutionShape.padTop;
    int padLeft = convolutionShape.padLeft;
    
    real4 in[9][9];
    real4 ker0[3][3],  ker1[3][3], ker2[3][3], ker3[3][3];
    
    real4 biasV = convolutionShape.haveBias ? ((device real4*)(bia + z))[0] : 0;
    
    real4 out[4][4] = {
        {biasV, biasV, biasV, biasV},
        {biasV, biasV, biasV, biasV},
        {biasV, biasV, biasV, biasV},
        {biasV, biasV, biasV, biasV}
    };
    
    int inputTop  = 2 * y - padTop;
    int inputLeft = 2 * x - padLeft;
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannel; c += 4) {
        /**get the kernel and input data*/
        BROU(GetVector4FromInput)(input, inputHeight, inputWidth, inputChannel, inputTop, inputLeft, c, (thread real4*)in, 9, 9);
        
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z    , c, ker0);
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 1, c, ker1);
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 2, c, ker2);
        BROU(GetKernel3X3Vector)(kerne, inputChannel, z + 3, c, ker3);
        
        for (int outY = 0, inY = 0; outY < maxOutY; ++outY, inY += 2) {
            for (int outX = 0, inX = 0; outX < maxOutX; ++outX, inX += 2) {
                out[outY][outX].x += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker3));
            }
        }
    }
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device real4 *outputV = (device real4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannel + z);
            
            outputV[0] = out[outY][outX];
        }
    }
}

#endif

















