/**
 * BrouRGBAImageConvertLayer
 * Created by yanyuanchi on 2017/7/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * this layer just convert the uint8 RGBA image to half image
 */

#import "BrouRGBAConvertLayer.h"

@interface BrouRGBAConvertLayer() {
    /**store the shape*/
    id<MTLBuffer> _shape;
    
    RGBAConvertType _convertType;
    
    /**the image's size is (height, width)*/
    int _height;
    int _width;
}

@end

@implementation BrouRGBAConvertLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                        height:(int)height
                         width:(int)width
                   convertType:(RGBAConvertType)convertType {
    self = [super initWithLabel:@"BrouRGBAConvertLayer"];
    
    if (!self) {
        return self;
    }
    
    NSAssert(height > 0 && width > 0, @"the height and width must >= 0");
    
    _height = height;
    _width  = width;
    
    _convertType = convertType;
    
    /**init dim buffer*/
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    TensorShape *shapeRef = (TensorShape *)_shape.contents;
    
    shapeRef->dim0 = height;
    shapeRef->dim1 = width;
    
    if (_convertType == UINT8_TO_HALF) {
        _functionName = @"brouConvertRGBAUInt8ToHalf";
    } else if (_convertType == HALF_TO_UINT8) {
        _functionName = @"brouConvertRGBAHalfToUInt8";
    }
    
    [self buildComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
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

- (int)getOutputBytes {
    if (_convertType == UINT8_TO_HALF) {
        return _height * _width * 4 * 2;
    } else if (_convertType == HALF_TO_UINT8) {
        return _height * _width * 4;
    } else {
        NSAssert(false, @"convert type error");
        return 0;
    }
}

- (int)getInputBytes {
    if (_convertType == UINT8_TO_HALF) {
        return _height * _width * 4;
    } else if (_convertType == HALF_TO_UINT8) {
        return _height * _width * 4 * 2;
    } else {
        NSAssert(false, @"convert type error");
        return 0;
    }
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input   offset:0 atIndex:0];
    [encoder setBuffer:output  offset:0 atIndex:1];
    [encoder setBuffer:_shape  offset:0 atIndex:2];

    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_width + 31) / 32,
                                                (_height + 15) / 16,
                                                1);
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

@end









