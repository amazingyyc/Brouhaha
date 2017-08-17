/**
 * add operate
 * Created by yanyuanchi on 2017/7/18.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the add operate is on n-dimensional array (n = 1, 2, 3)
 * support the Broadcasting rule, ref:https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
 */

#import "BrouLayer.h"

@interface BrouAddLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                        in1Dim:(NSArray<NSNumber*>*) in1Dim
                        in2Dim:(NSArray<NSNumber*>*) in2Dim;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   float32Bias:(void*)bias
                       biasDim:(NSArray<NSNumber*>*) biasDim
                         inDim:(NSArray<NSNumber*>*) inDim;

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                          input1:(id<MTLBuffer>)input1
                          input2:(id<MTLBuffer>)input2
                          output:(id<MTLBuffer>)output;

@end
