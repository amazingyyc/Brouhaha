/**
 * Created by yanyuanchi on 2017/6/18.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */


#import <Foundation/Foundation.h>

@interface BrouNet : NSObject

- (instancetype)init;

- (void)addLayer:(BrouLayer*)layer;

- (void)configWithDevice:(id<MTLDevice>)device;

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(void*)input
                     inputLength:(int)inputLength
                          output:(void*)output
                    outputLength:(int)outputLength;

@end
