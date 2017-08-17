/**
 * BrouTransposedConvolutionMMLayer.m
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/2.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import <UIKit/UIKit.h>

#import "BrouTransposedConvolutionLayer.h"

@interface BrouTransposedConvolutionMMLayer : BrouTransposedConvolutionLayer {

}

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)kernelData
                                 bias:(void*)biasData
                    originInputHeight:(int)originInputHeight
                     originInputWidth:(int)originInputWidth
                   originInputChannel:(int)originInputChannel
                   originOutputHeight:(int)originOutputHeight
                    originOutputWidth:(int)originOutputWidth
                  originOutputChannel:(int)originOutputChannel
                   originKernelHeight:(int)originKernelHeight
                    originKernelWidth:(int)originKernelWidth
                        originPadLeft:(int)originPadLeft
                         originPadTop:(int)originPadTop
                        originStrideX:(int)originStrideX
                        originStrideY:(int)originStrideY
                       outputAddRight:(int)outputAddRight
                      outputAddBottom:(int)outputAddBottom;

@end
