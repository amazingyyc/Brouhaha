/**
 * Created by yanyuanchi on 2017/7/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the batch Normalization layer
 */

#import "BrouLayer.h"

@interface BrouBatchNormalizationLayer : BrouLayer

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                                alpha:(void*)alpha
                                 beta:(void*)beta
                              epsilon:(float)epsilon
                               height:(int)height
                                width:(int)width
                              channel:(int)channel;

@end
