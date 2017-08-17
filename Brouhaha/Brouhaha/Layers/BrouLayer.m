/**
 * BrouLayer.h
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the protocol of all layers
 */
#import "BrouLayer.h"

@implementation BrouLayer

- (instancetype)initWithLabel:(NSString*)label {
    if (self = [super init]) {
        _label = label;
    }

    return self;
}

/**
 compute the operator

 @param commandBuffer the command buffer of mrtal
 @param input input buffer
 @param output output buffer
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {

}

/**
 * return the output bytes
 */
- (int)getOutputBytes {
    return 0;
}

/**
 * get the bytes of input
 */
- (int)getInputBytes {
    return 0;
}

@end
