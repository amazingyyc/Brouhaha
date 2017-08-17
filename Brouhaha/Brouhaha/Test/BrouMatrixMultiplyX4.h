/**
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * just for test
 */

#import "BrouLayer.h"

@interface BrouMatrixMultiplyX4 : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                             M:(int)M
                             K:(int)K
                             N:(int)N;

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                               A:(float*)A
                               B:(float*)B;

- (void)copyInA:(float*)A;
- (void)copyInB:(float*)B;
- (void)copyTOC:(float*)C;

@end
