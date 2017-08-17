/**
 * BrouReLuLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the ReLu layer
 */

#import "BrouUtils.h"
#import "BrouReLuLayer.h"

@interface BrouReLuLayer() {
    DimensionType _dimensionType;
    
    id<MTLBuffer> _shape;
    
    int _len;
}

@end

@implementation BrouReLuLayer

/**
 * ReLu
 * f(x) = x (x >= 0)
 * f(x) = 0 (x < 0)
 */
- (instancetype)initReLuWithDevice:(id<MTLDevice>)device
                           library:(id<MTLLibrary>)library
                            height:(int)height
                             width:(int)width
                           channel:(int)channel {
    self = [super initWithLabel:@"BrouReLuLayer"];

    if (!self) {
        return self;
    }

    int channelX4 = (channel + 3) / 4 * 4;
    
    _functionName  = @"brouReLu3D";
    _dimensionType = Dimension3D;
    _len = height * width * channelX4;
    
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    shapeRef->dim0 = height;
    shapeRef->dim1 = width;
    shapeRef->dim2 = channelX4;

    [self buildComputePipelinesStateWithDevice:device library:library];

    return self;
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
    /**get the function*/
    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:_functionName];

    if (!function) {
        NSLog(@"init  function error");

        return;
    }

    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];

    if (!_computePipelineState) {
        NSLog(@"init MTLComputePipelineState error");
    }
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];

    [encoder setBuffer:input  offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:_shape offset:0 atIndex:2];

    NSUInteger exeWidth = _computePipelineState.threadExecutionWidth;
    MTLSize threadsPerThreadgroup;
    MTLSize threadgroupsPerGrid;
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    
    if (_dimensionType == Dimension1D) {
        threadsPerThreadgroup = MTLSizeMake(exeWidth, 1, 1);
        threadgroupsPerGrid   = MTLSizeMake((shapeRef->dim0 + exeWidth * 4 - 1) / (exeWidth * 4),
                                            1,
                                            1);
    } else if (_dimensionType == Dimension2D) {
        threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
        threadgroupsPerGrid   = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                                            (shapeRef->dim0 + 15) / 16,
                                            1);
    } else if (_dimensionType == Dimension3D) {
        threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
        threadgroupsPerGrid   = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                                            (shapeRef->dim0 + 15) / 16,
                                            (shapeRef->dim2 / 4));
    } else {
        /**todo support all dimension data*/
        NSAssert(false, @"The data dimension is error");
        
        threadsPerThreadgroup = MTLSizeMake(0, 0, 0);
        threadgroupsPerGrid   = MTLSizeMake(0,
                                            0,
                                            0);
    }

    [encoder dispatchThreadgroups:threadgroupsPerGrid
            threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

/**
 * return the bytes of output
 */
- (int)getOutputBytes {
    return _len * 2;
}

- (int)getInputBytes {
    return _len * 2;
}

@end

















