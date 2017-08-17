/**
 * Created by yanyuanchi on 2017/6/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * convert float32 to float16 or float16 to float32
 */

#import "BrouLayer.h"

typedef NS_ENUM(NSInteger, ConvertType) {
    FLOAT32_TO_FLOAT16 = 0,
    FLOAT16_TO_FLOAT32 = 1,
};

@interface BrouConvertLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                        height:(int)height
                         width:(int)width
                       channel:(int)channel
                   convertType:(ConvertType)type;

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                       channel:(int)channel
                   convertType:(ConvertType)type;

@end
