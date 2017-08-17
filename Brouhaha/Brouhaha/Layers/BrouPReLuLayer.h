/**
 * BrouReLuLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the PReLu layer
 */

#import "BrouLayer.h"

@interface BrouPReLuLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)lirary
                        height:(int)height
                         width:(int)width
                       channel:(int)channel
                             a:(float)a;

@end
