/**
 * BrouLinearLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the linear layer
 */

#import <Foundation/Foundation.h>
#import "BrouLayer.h"

@interface BrouLinearLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)lirary
                        height:(int)height
                         width:(int)width
                       channel:(int)channel
                             a:(float)a
                             b:(float)b;

@end
