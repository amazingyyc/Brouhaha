/**
 * Created by yanyuanchi on 2017/5/30.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * use the matrix multiply to calculate the convolution
 */

#import "BrouLayer.h"
#import "BrouConvolutionLayer.h"

@interface BrouConvolutionMMLayer : BrouConvolutionLayer {
    /**the 3d input will convert to a matrix*/
    id<MTLBuffer> _inputMatrix;

    /**
     * the input and kernel will be convert to matrix
     * the input matrix's dimension is (MX4, K)
     * the kernel matrix's dimension is (K, NX4)
     * M = outputHeight*outputWidth
     * K = kernelHeight*kernelWidth*inputChannel
     * N = outputChannel
     */
    int _M;
    int _K;
    int _N;

    /**
     * MX4 >= M and timed by 4
     * NX4 >= N and timed by 4
     */
    int _MX4;
    int _NX4;

    /**the fucntion name of onvert input to a matrix*/
    NSString *_convertInputFunctionName;

    /**the pipe state*/
    id<MTLComputePipelineState> _convertInputComputePipelineState;
}

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)kernelData
                                 bias:(void*)biasData
                          inputHeight:(int)inputHeight
                           inputWidth:(int)inputWidth
                        intputChannel:(int)inputChannel
                         outputHeight:(int)outputHeight
                          outputWidth:(int)outputWidth
                        outputChannel:(int)outputChannel
                         kernelHeight:(int)kernelHeight
                          kernelWidth:(int)kernelWidth
                              padLeft:(int)padLeft
                               padTop:(int)padTop
                              strideX:(int)strideX
                              strideY:(int)strideY;

- (instancetype)initWithFloat16Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)float16Kernel
                                 bias:(void*)float16Bias
                          inputHeight:(int)inputHeight
                           inputWidth:(int)inputWidth
                        intputChannel:(int)inputChannel
                         outputHeight:(int)outputHeight
                          outputWidth:(int)outputWidth
                        outputChannel:(int)outputChannel
                         kernelHeight:(int)kernelHeight
                          kernelWidth:(int)kernelWidth
                              padLeft:(int)padLeft
                               padTop:(int)padTop
                              strideX:(int)strideX
                              strideY:(int)strideY;

- (void)configBufferWithDevice:(id<MTLDevice>)device kernel:(void*)float16Kernel;
- (void)configBufferWithDevice:(id<MTLDevice>)device bias:(void*)float16Bias;

@end
