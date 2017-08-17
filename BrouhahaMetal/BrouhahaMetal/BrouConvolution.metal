/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 * 
 * the convolution opetator
 */

#include <metal_stdlib>

using namespace metal;

/**the height width and channel of the input data, and the input date's dimension is (height, width, channel)*/
constant int inputHeight  [[function_constant(0)]];
constant int inputWidth   [[function_constant(1)]];
constant int inputChannel [[function_constant(2)]];

/**the output data and the diemsion is (height, width, channel)*/
constant int outputHeight  [[function_constant(3)]];
constant int outputWidth   [[function_constant(4)]];
constant int outputChannel [[function_constant(5)]];

/**the kernel data, the kernel's dimesion is (outputchannel, height, width, inputchannel)*/
constant int kernelHeight [[function_constant(6)]];
constant int kernelWidth  [[function_constant(7)]];

/**the pad of the input*/
constant int padLeft [[function_constant(8)]];
constant int padTop  [[function_constant(9)]];

/**the step of the kernel*/
constant int strideX [[function_constant(10)]];
constant int strideY [[function_constant(11)]];

/**
 * inputchannelx4 >= inputchannel and timed by 4
 * outputchannelx4 >= outputchannel and timed by 4
 */
constant int inputChannelX4  [[function_constant(12)]];
constant int outputChannelX4 [[function_constant(13)]];

/**
 * if the convolution has a bias
 */
constant bool haveBias[[function_constant(14)]];

/**
 * every thread deal with 4 X 4 X 4 output
 * the output is spilted to (4, 4, 4) blocks, and every thread handle one block
 * the input's dimension is (inputHeight, inputWidth, intputChannelX4)
 * the output's dimeansion is (outputHeight, outputWidth, outputChannelX4)
 * the weights's dimension is (outputChannelX4, kernelheight, kernelwidth, inputChannelX4)
 * the bias's dimesion is (outputchannleX4, 1), if have
 */
kernel void brouConvolution(device half *input   [[buffer(0)]],
                            device half *kerne   [[buffer(1)]],
                            device half *bia     [[buffer(2)]],
                            device half *output  [[buffer(3)]],
                            ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;

    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
    
    int maxOutY = min(y + 4, outputHeight);
    int maxOutX = min(x + 4, outputWidth);
    
    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
    
    for (int outY = y; outY < maxOutY; ++outY) {
        for (int outX = x; outX < maxOutX; ++outX) {
            half4 out = 0;
            
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
                    device half *inputOffset = input + (inY * inputWidth + inX) * inputChannelX4;
                    
                    device half *kernelOffset0 = kerne + (((z  ) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannelX4;
                    device half *kernelOffset1 = kerne + (((z+1) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannelX4;
                    device half *kernelOffset2 = kerne + (((z+2) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannelX4;
                    device half *kernelOffset3 = kerne + (((z+3) * kernelHeight + kernelY) * kernelWidth + kernelX) * inputChannelX4;
                    
                    for (int c = 0; c < inputChannelX4; c += 4) {
                        half4 inV = ((device half4*)(inputOffset))[0];
                        
                        half4 kernelV0 = ((device half4*)(kernelOffset0))[0];
                        half4 kernelV1 = ((device half4*)(kernelOffset1))[0];
                        half4 kernelV2 = ((device half4*)(kernelOffset2))[0];
                        half4 kernelV3 = ((device half4*)(kernelOffset3))[0];

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
            
            device half4 *outputV = (device half4*)(output + (outY * outputWidth + outX) * outputChannelX4 + z);
            
            outputV[0] = out + biasV;
        }
    }
}

#define VECTOR_DOT(a, offsetY, offsetX, b, y, x) dot(a[offsetY+y][offsetX+x], b[y][x])

#define ROW_VECTOR_DOT_COL0(a, offsetY, offsetX, b, row) VECTOR_DOT(a, offsetY, offsetX, b, row, 0)
#define ROW_VECTOR_DOT_COL1(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL0(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 1)
#define ROW_VECTOR_DOT_COL2(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL1(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 2)
#define ROW_VECTOR_DOT_COL3(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL2(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 3)
#define ROW_VECTOR_DOT_COL4(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL3(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 4)
#define ROW_VECTOR_DOT_COL5(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL4(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 5)
#define ROW_VECTOR_DOT_COL6(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL5(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 6)
#define ROW_VECTOR_DOT_COL7(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL6(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 7)
#define ROW_VECTOR_DOT_COL8(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL7(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 8)

#define COL_VECTOR_DOT_ROW0(a, offsetY, offsetX, b, col) VECTOR_DOT(a, offsetY, offsetX, b, 0, col)
#define COL_VECTOR_DOT_ROW1(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW0(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 1, col)
#define COL_VECTOR_DOT_ROW2(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW1(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 2, col)
#define COL_VECTOR_DOT_ROW3(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW2(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 3, col)
#define COL_VECTOR_DOT_ROW4(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW3(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 4, col)
#define COL_VECTOR_DOT_ROW5(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW4(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 5, col)
#define COL_VECTOR_DOT_ROW6(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW5(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 6, col)
#define COL_VECTOR_DOT_ROW7(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW6(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 7, col)
#define COL_VECTOR_DOT_ROW8(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW7(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 8, col)

#define MATRIX_VECTOR_DOT_0X0(a, offsetY, offsetX, b) VECTOR_DOT(a, offsetY, offsetX, b, 0, 0)
#define MATRIX_VECTOR_DOT_1X1(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_0X0(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL0(a, offsetY, offsetX, b, 1) \
                                                    + COL_VECTOR_DOT_ROW0(a, offsetY, offsetX, b, 1) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 1, 1)

#define MATRIX_VECTOR_DOT_2X2(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_1X1(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL1(a, offsetY, offsetX, b, 2) \
                                                    + COL_VECTOR_DOT_ROW1(a, offsetY, offsetX, b, 2) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 2, 2)

#define MATRIX_VECTOR_DOT_3X3(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_2X2(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL2(a, offsetY, offsetX, b, 3) \
                                                    + COL_VECTOR_DOT_ROW2(a, offsetY, offsetX, b, 3) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 3, 3)

#define MATRIX_VECTOR_DOT_4X4(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_3X3(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL3(a, offsetY, offsetX, b, 4) \
                                                    + COL_VECTOR_DOT_ROW3(a, offsetY, offsetX, b, 4) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 4, 4)

#define MATRIX_VECTOR_DOT_5X5(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_4X4(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL4(a, offsetY, offsetX, b, 5) \
                                                    + COL_VECTOR_DOT_ROW4(a, offsetY, offsetX, b, 5) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 5, 5)

#define MATRIX_VECTOR_DOT_6X6(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_5X5(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL5(a, offsetY, offsetX, b, 6) \
                                                    + COL_VECTOR_DOT_ROW5(a, offsetY, offsetX, b, 6) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 6, 6)

#define MATRIX_VECTOR_DOT_7X7(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_6X6(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL6(a, offsetY, offsetX, b, 7) \
                                                    + COL_VECTOR_DOT_ROW6(a, offsetY, offsetX, b, 7) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 7, 7)

#define MATRIX_VECTOR_DOT_8X8(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_7X7(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL7(a, offsetY, offsetX, b, 8) \
                                                    + COL_VECTOR_DOT_ROW7(a, offsetY, offsetX, b, 8) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 8, 8)

/**
 * for special kernel/stride use the special kernel function
 */
inline void getInputVectorHalf4(device half *input,
                                int height, int width, int channel,
                                int y, int x, int z,
                                thread half4 *vector,
                                int row, int col) {
    for (int j = 0; j < row; ++j) {
        for (int i = 0; i < col; ++i) {
            if ((y + j) >= height || (x + i) >= width || z >= channel) {
                vector[j * col + i] = 0;
            } else {
                device half4 *inputV = (device half4*)(input + ((y + j) * width + (x + i)) * channel + z);
                vector[j * col + i] = inputV[0];
            }
        }
    }
}

/**
 * the kernel is 3X3
 */
inline void getKernel3X3VectorHalf4(device half *kerne, int inputChannel, int outputOffset, int channelOffset, half4 vector[3][3]) {
    device half *kernelOffset = kerne + outputOffset * 3 * 3 * inputChannel + channelOffset;
    
    vector[0][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[1][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[2][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][2] = ((device half4*)(kernelOffset))[0];
}

kernel void brouConvolutionKernel3X3(device half *input   [[buffer(0)]],
                                     device half *kerne   [[buffer(1)]],
                                     device half *bia     [[buffer(2)]],
                                     device half *output  [[buffer(3)]],
                                     ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
    
    half4 in[3][3];
    half4 ker0[3][3],  ker1[3][3], ker2[3][3], ker3[3][3];
    
    half4 out[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannelX4; c += 4) {
        /**get the kernel and input data*/
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z    , c, ker0);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 1, c, ker1);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 2, c, ker2);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 3, c, ker3);
        
        for (int outY = 0; outY < maxOutY; ++outY) {
            for (int outX = 0; outX < maxOutX; ++outX) {
                int inputTop  = -padTop  + strideY * (y + outY);
                int inputLeft = -padLeft + strideX * (x + outX);
                
                getInputVectorHalf4(input, inputHeight, inputWidth, inputChannelX4, inputTop, inputLeft, c, (thread half4*)in, 3, 3);
                
                out[outY][outX].x += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_2X2(in, 0, 0, ker3));
            }
        }
    }
    
    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device half4 *outputV = (device half4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannelX4 + z);
            outputV[0] = out[outY][outX] + biasV;
        }
    }
}

kernel void brouConvolutionKernel3X3Stride1X1(device half *input   [[buffer(0)]],
                                              device half *kerne   [[buffer(1)]],
                                              device half *bia     [[buffer(2)]],
                                              device half *output  [[buffer(3)]],
                                              ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
    
    half4 in[6][6];
    half4 ker0[3][3],  ker1[3][3], ker2[3][3], ker3[3][3];
    
    half4 out[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    
    int inputTop  = y - padTop;
    int inputLeft = x - padLeft;
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannelX4; c += 4) {
        /**get the kernel and input data*/
        getInputVectorHalf4(input, inputHeight, inputWidth, inputChannelX4, inputTop, inputLeft, c, (thread half4*)in, 6, 6);
        
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z    , c, ker0);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 1, c, ker1);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 2, c, ker2);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 3, c, ker3);
        
        for (int outY = 0, inY = 0; outY < maxOutY; ++outY, ++inY) {
            for (int outX = 0, inX = 0; outX < maxOutX; ++outX, ++inX) {
                out[outY][outX].x += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker3));
            }
        }
    }
    
    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device half4 *outputV = (device half4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannelX4 + z);
            outputV[0] = out[outY][outX] + biasV;
        }
    }
}

kernel void brouConvolutionKernel3X3Stride2X2(device half *input   [[buffer(0)]],
                                              device half *kerne   [[buffer(1)]],
                                              device half *bia     [[buffer(2)]],
                                              device half *output  [[buffer(3)]],
                                              ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
    
    half4 in[9][9];
    half4 ker0[3][3],  ker1[3][3], ker2[3][3], ker3[3][3];
    
    half4 out[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    
    int inputTop  = 2 * y - padTop;
    int inputLeft = 2 * x - padLeft;
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannelX4; c += 4) {
        /**get the kernel and input data*/
        getInputVectorHalf4(input, inputHeight, inputWidth, inputChannelX4, inputTop, inputLeft, c, (thread half4*)in, 9, 9);
        
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z    , c, ker0);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 1, c, ker1);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 2, c, ker2);
        getKernel3X3VectorHalf4(kerne, inputChannelX4, z + 3, c, ker3);
        
        for (int outY = 0, inY = 0; outY < maxOutY; ++outY, inY += 2) {
            for (int outX = 0, inX = 0; outX < maxOutX; ++outX, inX += 2) {
                out[outY][outX].x += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_2X2(in, inY, inX, ker3));
            }
        }
    }
    
    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device half4 *outputV = (device half4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannelX4 + z);
            outputV[0] = out[outY][outX] + biasV;
        }
    }
}

/**
 * the kernel is 5X5
 */
inline void getKernel5X5VectorHalf4(device half *kerne, int inputChannel, int outputOffset, int channelOffset, half4 vector[5][5]) {
    device half *kernelOffset = kerne + outputOffset * 5 * 5 * inputChannel + channelOffset;
    
    vector[0][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[0][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[1][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[1][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[2][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[2][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[3][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[3][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[3][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[3][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[3][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    
    vector[4][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[4][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[4][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[4][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
    vector[4][4] = ((device half4*)(kernelOffset))[0];
}

kernel void brouConvolutionKernel5X5(device half *input   [[buffer(0)]],
                                     device half *kerne   [[buffer(1)]],
                                     device half *bia     [[buffer(2)]],
                                     device half *output  [[buffer(3)]],
                                     ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
    
    half4 in[5][5];
    half4 ker0[5][5],  ker1[5][5], ker2[5][5], ker3[5][5];
    
    half4 out[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannelX4; c += 4) {
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z    , c, ker0);
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z + 1, c, ker1);
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z + 2, c, ker2);
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z + 3, c, ker3);
        
        for (int outY = 0; outY < maxOutY; ++outY) {
            for (int outX = 0; outX < maxOutX; ++outX) {
                int inputTop  = -padTop  + strideY * (y + outY);
                int inputLeft = -padLeft + strideX * (x + outX);
                
                getInputVectorHalf4(input, inputHeight, inputWidth, inputChannelX4, inputTop, inputLeft, c, (thread half4*)in, 5, 5);
                
                out[outY][outX].x += (MATRIX_VECTOR_DOT_4X4(in, 0, 0, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_4X4(in, 0, 0, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_4X4(in, 0, 0, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_4X4(in, 0, 0, ker3));
            }
        }
    }
    
    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device half4 *outputV = (device half4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannelX4 + z);
            outputV[0] = out[outY][outX] + biasV;
        }
    }
}

kernel void brouConvolutionKernel5X5Stride1X1(device half *input   [[buffer(0)]],
                                              device half *kerne   [[buffer(1)]],
                                              device half *bia     [[buffer(2)]],
                                              device half *output  [[buffer(3)]],
                                              ushort3 grid [[thread_position_in_grid]]) {
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
        return;
    }
    
    half4 in[8][8];
    half4 ker0[5][5],  ker1[5][5], ker2[5][5], ker3[5][5];
    
    half4 out[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    
    int inputTop  = y - padTop;
    int inputLeft = x - padLeft;
    
    int maxOutY = min(4, outputHeight - y);
    int maxOutX = min(4, outputWidth  - x);
    
    for (int c = 0; c < inputChannelX4; c += 4) {
        /**get the kernel and input data*/
        getInputVectorHalf4(input, inputHeight, inputWidth, inputChannelX4, inputTop, inputLeft, c, (thread half4*)in, 8, 8);
        
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z    , c, ker0);
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z + 1, c, ker1);
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z + 2, c, ker2);
        getKernel5X5VectorHalf4(kerne, inputChannelX4, z + 3, c, ker3);
        
        for (int outY = 0, inY = 0; outY < maxOutY; ++outY, ++inY) {
            for (int outX = 0, inX = 0; outX < maxOutX; ++outX, ++inX) {
                out[outY][outX].x += (MATRIX_VECTOR_DOT_4X4(in, inY, inX, ker0));
                out[outY][outX].y += (MATRIX_VECTOR_DOT_4X4(in, inY, inX, ker1));
                out[outY][outX].z += (MATRIX_VECTOR_DOT_4X4(in, inY, inX, ker2));
                out[outY][outX].w += (MATRIX_VECTOR_DOT_4X4(in, inY, inX, ker3));
            }
        }
    }
    
    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
    
    for (int outY = 0; outY < maxOutY; ++outY) {
        for (int outX = 0; outX < maxOutX; ++outX) {
            device half4 *outputV = (device half4*)(output + ((y + outY) * outputWidth + x + outX) * outputChannelX4 + z);
            outputV[0] = out[outY][outX] + biasV;
        }
    }
}

/**
 * the kernel is 7X7
 */
//inline void getKernel7X7VectorHalf4(device half *kerne, int inputChannel, int outputOffset, int channelOffset, half4 vector[7][7]) {
//    device half *kernelOffset = kerne + outputOffset * 7 * 7 * inputChannel + channelOffset;
//
//    vector[0][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[1][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[2][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[3][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[4][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[5][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[6][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][6] = ((device half4*)(kernelOffset))[0];
//}

/**
 * the kernel is 9X9
 */
//inline void getKernel9X9VectorHalf4(device half *kerne, int inputChannel, int outputOffset, int channelOffset, half4 vector[9][9]) {
//    device half *kernelOffset = kerne + outputOffset * 9 * 9 * inputChannel + channelOffset;
//
//    vector[0][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[0][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[1][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[1][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[2][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[2][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[3][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[3][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[4][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[4][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[5][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][6] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[5][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[6][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[6][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[7][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[7][8] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//
//    vector[8][0] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][1] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][2] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][3] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][4] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][5] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][7] = ((device half4*)(kernelOffset))[0]; kernelOffset += inputChannel;
//    vector[8][8] = ((device half4*)(kernelOffset))[0];
//}
//
//kernel void brouConvolutionKernel9X9(device half *input   [[buffer(0)]],
//                                     device half *kerne   [[buffer(1)]],
//                                     device half *bia     [[buffer(2)]],
//                                     device half *output  [[buffer(3)]],
//                                     ushort3 grid [[thread_position_in_grid]]) {
//    int x = grid.x << 2;
//    int y = grid.y << 2;
//    int z = grid.z << 2;
//
//    if (x >= outputWidth || y >= outputHeight || z >= outputChannel) {
//        return;
//    }
//
//    half4 in[9][9];
//    half4 ker0[9][9], ker1[9][9], ker2[9][9], ker3[9][9];
//
//    half4 out[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
//
//    int maxOutY = min(4, outputHeight - y);
//    int maxOutX = min(4, outputWidth  - x);
//
//    for (int c = 0; c < inputChannelX4; c += 4) {
//        getKernel9X9VectorHalf4(kerne, inputChannelX4, z  , c, ker0);
//        getKernel9X9VectorHalf4(kerne, inputChannelX4, z+1, c, ker1);
//        getKernel9X9VectorHalf4(kerne, inputChannelX4, z+2, c, ker2);
//        getKernel9X9VectorHalf4(kerne, inputChannelX4, z+3, c, ker3);
//
//        for (int outY = 0; outY < maxOutY; ++outY) {
//            for (int outX = 0; outX < maxOutX; ++outX) {
//                int inputTop  = -padTop  + strideY * (y + outY);
//                int inputLeft = -padLeft + strideX * (x + outX);
//
//                getInputVectorHalf4(input, inputHeight, inputWidth, inputChannelX4, inputTop, inputLeft, c, (thread half4*)in, 9, 9);
//
//                out[outY][outX].x += (MATRIX_VECTOR_DOT_8X8(in, 0, 0, ker0));
//                out[outY][outX].y += (MATRIX_VECTOR_DOT_8X8(in, 0, 0, ker1));
//                out[outY][outX].z += (MATRIX_VECTOR_DOT_8X8(in, 0, 0, ker2));
//                out[outY][outX].w += (MATRIX_VECTOR_DOT_8X8(in, 0, 0, ker3));
//            }
//        }
//    }
//
//    half4 biasV = haveBias ? ((device half4*)(bia + z))[0] : 0;
//
//    for (int outY = 0; outY < maxOutY; ++outY) {
//        for (int outX = 0; outX < maxOutX; ++outX) {
//            device half4 *outputV = (device half4*)(output + ((y + outY) * outputWidth + (x + outX)) * outputChannelX4 + z);
//
//            outputV[0] = out[outY][outX] + biasV;
//        }
//    }
//}






