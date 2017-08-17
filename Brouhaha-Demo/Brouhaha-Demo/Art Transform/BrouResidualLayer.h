/**
 * Created by yanyuanchi on 2017/7/23.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * a residual layer
 * include two convolution layer, two batch-norm layer, two relu layer
 */

#import "BrouLayer.h"

@interface BrouResidualLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                float32Weight1:(void*)weight1
                float32Weight2:(void*)weight2
                 float32Alpha1:(void*)alpha1
                  float32Beta1:(void*)beta1
                 float32Alpha2:(void*)alpha2
                  float32Beta2:(void*)beta2
                        height:(int)height
                         width:(int)width;
@end
