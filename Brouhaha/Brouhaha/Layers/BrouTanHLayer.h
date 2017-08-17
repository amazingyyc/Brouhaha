/**
 * BrouTanHLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the TanH layer
 */

#import <Foundation/Foundation.h>
#import "BrouLayer.h"

@interface BrouTanHLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)lirary
                       channel:(int)channel;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)lirary
                        height:(int)height
                         width:(int)width
                       channel:(int)channel;

@end
