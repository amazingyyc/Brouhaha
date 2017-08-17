/**
 * Created by yanyuanchi on 2017/7/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the batch Normalization layer
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouBatchNormalizationLayer.h"

@interface BrouBatchNormalizationLayer() {
    /**
     * in testing the mean and variance will be knowed
     * if don't the mean and variance will be calculate by
     * brouCalculateMeanAndVariance3D
     *
     * the alpha and beta is knowed
     *
     * output = alpha * (input - mean) / (sqrt(variance + epsilon)) + beta
     */
    
    /**
     * if the input and the output is 3d
     * then the input ant output's dimension is (height, width, channelX4)
     *
     * if the input and output is 1d than the dimension is (channelX4, 1)
     *
     * the mean and variance's dimension is (channelX4, 1)
     */
    DimensionType _dimensionType;
    
    int _height;
    int _width;
    int _channel;
    int _channelX4;
    
    float _epsilon;
    
    id<MTLBuffer> _mean;
    id<MTLBuffer> _variance;
    id<MTLBuffer> _alpha;
    id<MTLBuffer> _beta;
    
    NSString *_calculateMeanAndVarianceFunctionName;
    
    id<MTLComputePipelineState> _calculateMeanAndVariancePipelineState;
}

@end

@implementation BrouBatchNormalizationLayer

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                                alpha:(void*)alpha
                                 beta:(void*)beta
                              epsilon:(float)epsilon
                               height:(int)height
                                width:(int)width
                              channel:(int)channel {
    self = [super initWithLabel:@"BrouBatchNormalizationLayer"];
    
    if (!self) {
        return self;
    }
    
    _height  = height;
    _width   = width;
    _channel = channel;
    
    _channelX4 = (_channel + 3) / 4 * 4;
    
    _dimensionType = Dimension3D;
    
    _epsilon = epsilon;
    
    /**init mtlbuffer*/
    _mean = [device newBufferWithLength:2 * _channelX4
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _variance = [device newBufferWithLength:2 * _channelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _alpha = [device newBufferWithLength:2 * _channelX4
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _beta = [device newBufferWithLength:2 * _channelX4
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    uint16_t *float16Data = (uint16_t*)malloc(2 * _channel);

    /**alpha*/
    convertFloat32ToFloat16(alpha, float16Data, _channel);
    memset(_alpha.contents, 0, 2 * _channelX4);
    memcpy(_alpha.contents, float16Data, 2 * _channel);
    
    /**beta*/
    convertFloat32ToFloat16(beta, float16Data, _channel);
    memset(_beta.contents, 0, 2 * _channelX4);
    memcpy(_beta.contents, float16Data, 2 * _channel);

    free(float16Data);
    
    _functionName                         = @"brouBatchNormalization3D";
    _calculateMeanAndVarianceFunctionName = @"brouCalculateMeanAndVariance3D";
    
    [self buildComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
    /**set the function constant*/
    MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
    [constantValues setConstantValue:&_height     type:MTLDataTypeInt   atIndex:0];
    [constantValues setConstantValue:&_width      type:MTLDataTypeInt   atIndex:1];
    [constantValues setConstantValue:&_channel    type:MTLDataTypeInt   atIndex:2];
    [constantValues setConstantValue:&_channelX4  type:MTLDataTypeInt   atIndex:3];
    [constantValues setConstantValue:&_epsilon    type:MTLDataTypeFloat atIndex:4];
    
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
        
        return;
    }
    
    /**calculate mean and variance*/
    function = [library newFunctionWithName:_calculateMeanAndVarianceFunctionName
                             constantValues:constantValues error:&error];
    
    if (!function) {
        NSLog(@"init _calculateMeanAndVarianceFunction fail!");
        
        return;
    }
    
    _calculateMeanAndVariancePipelineState = [device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
    
    if (!_calculateMeanAndVariancePipelineState) {
        NSLog(@"init _calculateMeanAndVariancePipelineState error");
    }
}

- (int)getInputBytes {
    return _height * _width * _channelX4 * 2;
}

- (int)getOutputBytes {
    return _height * _width * _channelX4 * 2;
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    /**calcualte mean*/
    id<MTLComputeCommandEncoder> meanAndVarianceEncoder = [commandBuffer computeCommandEncoder];
    [meanAndVarianceEncoder setComputePipelineState:_calculateMeanAndVariancePipelineState];
    [meanAndVarianceEncoder setBuffer:input     offset:0 atIndex:0];
    [meanAndVarianceEncoder setBuffer:_mean     offset:0 atIndex:1];
    [meanAndVarianceEncoder setBuffer:_variance offset:0 atIndex:2];
    
    NSUInteger exeWidth = _calculateMeanAndVariancePipelineState.threadExecutionWidth;
    
    MTLSize threadsPerThreadgroup = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_channel + exeWidth * 4 - 1) / (exeWidth * 4),
                                                1,
                                                1);
    
    [meanAndVarianceEncoder dispatchThreadgroups:threadgroupsPerGrid
                           threadsPerThreadgroup:threadsPerThreadgroup];
    [meanAndVarianceEncoder endEncoding];
    
    /**calcualte output*/
    id<MTLComputeCommandEncoder> bnEncoder = [commandBuffer computeCommandEncoder];
    [bnEncoder setComputePipelineState:_computePipelineState];
    [bnEncoder setBuffer:input     offset:0 atIndex:0];
    [bnEncoder setBuffer:output    offset:0 atIndex:1];
    [bnEncoder setBuffer:_mean     offset:0 atIndex:2];
    [bnEncoder setBuffer:_variance offset:0 atIndex:3];
    [bnEncoder setBuffer:_alpha    offset:0 atIndex:4];
    [bnEncoder setBuffer:_beta     offset:0 atIndex:5];
    
    threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    threadgroupsPerGrid   = MTLSizeMake((_width  + 31) / 32,
                                        (_height + 15) / 16,
                                        _channelX4 / 4);
    
    [bnEncoder dispatchThreadgroups:threadgroupsPerGrid
              threadsPerThreadgroup:threadsPerThreadgroup];
    [bnEncoder endEncoding];
}
@end

















