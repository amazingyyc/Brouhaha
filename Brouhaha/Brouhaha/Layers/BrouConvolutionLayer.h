/**
 * BrouLayer.h
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the convolution layer
 */

#import <Foundation/Foundation.h>
#import "BrouLayer.h"

/**
 * uae a struct to store the params of a convolution
 */
typedef struct _ConvolutionShape {
    /**the kernel size*/
    int32_t kernelHeight;
    int32_t kernelWidth;
    
    /**the pad of input*/
    int32_t padTop;
    int32_t padLeft;
    
    /**the stride of kernel, for transposed convolution always be 1*/
    int32_t strideY;
    int32_t strideX;
    
    /**the 0 units inserted to input of transposed convolution*/
    int32_t insertY;
    int32_t insertX;
    
    /**for dilated convolution*/
    int32_t dilatedY;
    int32_t dilatedX;
    
    /**if the convoluton has bias, 0 false, !0 true*/
    bool haveBias;
} ConvolutionShape;

@interface BrouConvolutionLayer : BrouLayer {
    /**MTLBUffer to store the kernel and bias data*/
    id<MTLBuffer> _kernel;
    id<MTLBuffer> _bias;

    /**ths input dimension is (inputHeight, inputWidth, intputChannel)*/
    int _inputHeight;
    int _inputWidth;
    int _inputChannel;

    /**the output diemsnion is (outputHeight, outputWidth, outputChannel)*/
    int _outputHeight;
    int _outputWidth;
    int _outputChannel;

    /**the kernel dimesnion is (outputChannel, kernelHeight, kernelWidth, inputChannel)*/
    int _kernelHeight;
    int _kernelWidth;

    /**the pad*/
    int _padLeft;
    int _padTop;

    /**stride of kernel*/
    int _strideX;
    int _strideY;

    /**
     * _inputChannelX4 >= inputchannel and timed by 4
     * _inputChannelX4 >= outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;

    /**if the convolution has a bias*/
    bool _haveBias;
}

- (void)configParametersInputHeight:(int)inputHeight
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

/**copy float16 data to MTL buffer*/
- (void)copyBufferWithKernel:(void*)float16Kernel;
- (void)copyBufferWithBias:(void*)float16Bias;

@end












