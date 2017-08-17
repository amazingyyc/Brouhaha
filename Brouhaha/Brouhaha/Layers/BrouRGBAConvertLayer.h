/**
 * BrouRGBAImageConvertLayer
 * Created by yanyuanchi on 2017/7/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * this layer just convert the uint8 RGBA image to half image
 */

#import "BrouLayer.h"

/**
 * convert type
 * uint8_t to half or half to uint8_t
 */
typedef NS_ENUM(NSInteger, RGBAConvertType) {
    UINT8_TO_HALF = 1,
    HALF_TO_UINT8 = 2
};

@interface BrouRGBAConvertLayer : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                        height:(int)height
                         width:(int)width
                   convertType:(RGBAConvertType)convertType;

@end
