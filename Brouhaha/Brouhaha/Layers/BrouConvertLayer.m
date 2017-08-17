/**
 * Created by yanyuanchi on 2017/6/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * convert float32 to float16 or float16 to float32
 */

#import "BrouConvertLayer.h"

@interface BrouConvertLayer() {
    int _height;
    int _width;
    int _channel;
    int _channelX4;

    ConvertType _convertType;
    DimensionType _dimensionType;
}

@end

@implementation BrouConvertLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                        height:(int)height
                         width:(int)width
                       channel:(int)channel
                   convertType:(ConvertType)type {
    self = [super initWithLabel:@"BrouConvertLayer"];

    if (self) {
        _height    = height;
        _width     = width;
        _channel   = channel;
        _channelX4 = (channel + 3) / 4;

        _convertType = type;
        _dimensionType = Dimension3D;

        [self buildComputePipelinesStateWithDevice:device library:library];
    }

    return self;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                       channel:(int)channel
                   convertType:(ConvertType)type {
    self = [super initWithLabel:@"BrouConvertLayer"];

    if (self) {
        _channel = channel;
        _channelX4 = (channel + 3) / 4;

        _convertType = type;
        _dimensionType = Dimension1D;

        [self buildComputePipelinesStateWithDevice:device library:library];
    }

    return self;
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
    if (_dimensionType == Dimension1D) {
        if (_convertType == FLOAT32_TO_FLOAT16) {
            _functionName = @"convertFloatToHalf1D";
        } else if (_convertType == FLOAT16_TO_FLOAT32) {
            _functionName = @"convertHalfToFloat1D";
        }
    } else if (_dimensionType == Dimension3D) {
        if (_convertType == FLOAT32_TO_FLOAT16) {
            _functionName = @"convertFloatToHalf3D";
        } else if (_convertType == FLOAT16_TO_FLOAT32) {
            _functionName = @"convertHalfToFloat3D";
        }
    }

    /**set the function constant*/
    MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];

    [constantValues setConstantValue:&_height     type:MTLDataTypeInt atIndex:0];
    [constantValues setConstantValue:&_width      type:MTLDataTypeInt atIndex:1];
    [constantValues setConstantValue:&_channel    type:MTLDataTypeInt atIndex:2];
    [constantValues setConstantValue:&_channelX4  type:MTLDataTypeInt atIndex:3];

    /**get the function*/
    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:_functionName constantValues:constantValues error:&error];

    if (!function) {
        NSLog(@"init  function error");

        return;
    }

    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];

    if (!_computePipelineState) {
        NSLog(@"init MTLComputePipelineState error");
    }
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
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];

    [encoder setBuffer:input  offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];

    NSUInteger exeWidth = _computePipelineState.threadExecutionWidth;
    MTLSize threadsPerThreadgroup;
    MTLSize threadgroupsPerGrid;

    if (_dimensionType == Dimension1D) {
        threadsPerThreadgroup = MTLSizeMake(4, 1, 1);
        threadgroupsPerGrid   = MTLSizeMake((_channelX4 + exeWidth * 4 - 1) / (exeWidth * 4),
                                            1, 1);
    } else {
        threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
        threadgroupsPerGrid   = MTLSizeMake((_width + 31) / 32,
                                            (_height + 15) / 16,
                                            _channelX4 / 4);
    }

    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

/**
 * return the bytes of output
 */
- (int)getOutputBytes {
    if (_dimensionType == Dimension1D) {
        if (_convertType == FLOAT32_TO_FLOAT16) {
            return _channelX4 * 2;
        } else if (_convertType == FLOAT16_TO_FLOAT32) {
            return _channelX4 * 4;
        }
    } else if (_dimensionType == Dimension3D) {
        if (_convertType == FLOAT32_TO_FLOAT16) {
            return _height * _width * _channelX4 * 2;
        } else if (_convertType == FLOAT16_TO_FLOAT32) {
            return _height * _width * _channelX4 * 4;
        }
    }

    return 0;
}

- (int)getInputBytes {
    if (_dimensionType == Dimension1D) {
        if (_convertType == FLOAT32_TO_FLOAT16) {
            return _channelX4 * 4;
        } else if (_convertType == FLOAT16_TO_FLOAT32) {
            return _channelX4 * 2;
        }
    } else if (_dimensionType == Dimension3D) {
        if (_convertType == FLOAT32_TO_FLOAT16) {
            return _height * _width * _channelX4 * 4;
        } else if (_convertType == FLOAT16_TO_FLOAT32) {
            return _height * _width * _channelX4 * 2;
        }
    }

    return 0;
}

@end









