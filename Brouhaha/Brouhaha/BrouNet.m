/**
 * Created by yanyuanchi on 2017/6/18.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import "BrouLayer.h"
#import "BrouNet.h"

@interface BrouNet() {
    NSMutableArray<BrouLayer*> *_layers;

    id<MTLBuffer> _inputBuffer;
    id<MTLBuffer> _outputBuffer;
}

@end

@implementation BrouNet

- (instancetype)init {
    self = [super init];

    if (self) {
        _layers = [[NSMutableArray alloc] init];
    }

    return self;
}

- (void)addLayer:(BrouLayer*)layer {
    if (layer) {
        [_layers addObject:layer];
    }
}

- (void)configWithDevice:(id<MTLDevice>)device {
    assert(0 != _layers.count);

    int maxByteCount = [_layers[0] getInputBytes];

    for (BrouLayer *layer in _layers) {
        maxByteCount = MAX(maxByteCount, [layer getOutputBytes]);
    }

    _inputBuffer  = [device newBufferWithLength:maxByteCount
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    _outputBuffer = [device newBufferWithLength:maxByteCount
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(void*)input
                     inputLength:(int)inputLength
                          output:(void*)output
                    outputLength:(int)outputLength {
    assert(0 != _layers.count);

    /**copy input*/
    memcpy(_inputBuffer.contents, input, inputLength);

    id<MTLBuffer> buffer0 = _inputBuffer;
    id<MTLBuffer> buffer1 = _outputBuffer;

    for (BrouLayer *layer in _layers) {
        [layer computeWithCommandBuffer:commandBuffer input:buffer0 output:buffer1];

        id<MTLBuffer> t = buffer0;
        buffer0 = buffer1;
        buffer1 = t;
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    /**copy to output*/
    memcpy(output, buffer0.contents, outputLength);
}

@end








