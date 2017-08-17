/**
 * BrouDilatedConvolutionMMLayer.h
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/2.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import <UIKit/UIKit.h>

#import "BrouConvolutionMMLayer.h"

@interface BrouDilatedConvolutionMMLayer : BrouConvolutionMMLayer {
    /**the dilated params on x/y axis*/
    int _dilatedX;
    int _dilatedY;
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
                              strideY:(int)strideY
                             dilatedX:(int)dilatedX
                             dilatedY:(int)dilatedY;

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
                              strideY:(int)strideY
                             dilatedX:(int)dilatedX
                             dilatedY:(int)dilatedY;

@end
