/**
 * BrouReLuLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the ReLu layer
 */

#import <Foundation/Foundation.h>
#import "BrouLayer.h"

@interface BrouReLuLayer : BrouLayer

- (instancetype)initReLuWithDevice:(id<MTLDevice>)device
                           library:(id<MTLLibrary>)library
                            height:(int)height
                             width:(int)width
                           channel:(int)channel;

@end
