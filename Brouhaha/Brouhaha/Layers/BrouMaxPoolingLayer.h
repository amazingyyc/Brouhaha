/**
 * BrouMaxPoolingLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the max pooling layer
 */

#import "BrouPoolingLayer.h"

@interface BrouMaxPoolingLayer : BrouPoolingLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   inputHeight:(int)inputHeight
                    inputWidth:(int)inputWidth
                  outputHeight:(int)outputHeight
                   outputWidth:(int)outputWidth
                       channel:(int)channel
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kerneWidth
                       padLeft:(int)padLeft
                        padTop:(int)padTop
                       strideX:(int)strideX
                       strideY:(int)strideY;

@end
