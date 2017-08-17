/**
 * BrouFullConnectLayer.h
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the fullconnect layer
 */

@import Metal;
@import MetalKit;

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouLayer.h"
#import "BrouFullConnectLayer.h"

@interface BrouFullConnectLayer() {
    /**
     * the parameters of fullconnect, output = kernel * input + bias
     *
     * the kernel's dimensio will be (outputChannelX4, inputChannelX4)
     * the bias's dimension is (outputChannelX4, 1)
     */
    id<MTLBuffer> _weights;
    id<MTLBuffer> _bias;

    int _inputChannel;
    int _outputChannel;

    /**
     * the inputChannleX4  great than or equal to inputChannel and timed by 4
     * the outputChannelX4 great than or equal to outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;

    BOOL _haveBias;
}

@end

@implementation BrouFullConnectLayer

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                              weights:(void*)weightsData
                                 bias:(void*)biasData
                        intputChannel:(int)inputChannel
                        outputChannel:(int)outputChannel {
    self = [super initWithLabel:@"BrouFullconnectLayer"];

    if (!self) {
        return self;
    }

    _inputChannel  = inputChannel;
    _outputChannel = outputChannel;

    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;

    if (biasData) {
        _haveBias = YES;
        _functionName = @"brouFullconnectBlock";

        void* float16Weights = (void*)malloc(sizeof(uint16_t) * inputChannel * outputChannel);
        void* float16Bias    = (void*)malloc(sizeof(uint16_t) * outputChannel);

        /**convert float32 to float16*/
        convertFloat32ToFloat16Two((uint32_t *)weightsData,
                                   (uint16_t *)float16Weights,
                                   inputChannel * outputChannel,
                                   (uint32_t *)biasData,
                                   (uint16_t *)float16Bias,
                                   outputChannel);

        _weights = [device newBufferWithLength:_inputChannelX4 * _outputChannelX4 * 2
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        _bias = [device newBufferWithLength:_outputChannelX4 * 2
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        [self copyDataWithWeights:float16Weights];
        [self copyDataWithBias:float16Bias];

        free(float16Weights);
        free(float16Bias);
    } else {
        _haveBias = NO;
        _functionName = @"brouFullconnectBlockWithoutBias";

        void* float16Weights = (void*)malloc(sizeof(uint16_t) * inputChannel * outputChannel);

        convertFloat32ToFloat16((uint32_t*)weightsData, (uint16_t*)float16Weights, inputChannel * outputChannel);

        _weights = [device newBufferWithLength:_inputChannelX4 * _outputChannelX4 * 2
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        [self copyDataWithWeights:float16Weights];

        free(float16Weights);
    }

    /**build the compute pipelinesstate*/
    [self buildComputePipelinesStateWithDevice:device library:library];

    return self;
}

- (instancetype)initWithFloat16Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                              weights:(void*)float16Weights
                                 bias:(void*)float16Bias
                        intputChannel:(int)inputChannel
                        outputChannel:(int)outputChannel {
    self = [super initWithLabel:@"BrouFullconnectLayer"];

    if (!self) {
        return self;
    }

    _inputChannel  = inputChannel;
    _outputChannel = outputChannel;

    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;

    if (float16Bias) {
        _haveBias = YES;
        _functionName = @"brouFullconnectBlock";

        _weights = [device newBufferWithLength:_inputChannelX4 * _outputChannelX4 * 2
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        _bias = [device newBufferWithLength:_outputChannelX4 * 2
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        [self copyDataWithWeights:float16Weights];
        [self copyDataWithBias:float16Bias];
    } else {
        _haveBias = NO;
        _functionName = @"brouFullconnectBlockWithoutBias";

        _weights = [device newBufferWithLength:_inputChannelX4 * _outputChannelX4 * 2
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        [self copyDataWithWeights:float16Weights];
    }

    /**build the compute pipelinesstate*/
    [self buildComputePipelinesStateWithDevice:device library:library];

    return self;
}

- (void)copyDataWithWeights:(void*)weigthsData {
    int weigthsDataRowByte   = _inputChannel   * 2;
    int weightsBufferRowByte = _inputChannelX4 * 2;

    void *weightsBufferData = _weights.contents;

    for (int i = 0; i < _outputChannel; ++i) {
        memset(weightsBufferData + weigthsDataRowByte, 0, (_inputChannelX4 - _inputChannel) * 2);
        memcpy(weightsBufferData, weigthsData, weigthsDataRowByte);

        weigthsData += weigthsDataRowByte;
        weightsBufferData += weightsBufferRowByte;
    }
}

- (void)copyDataWithBias:(void*)biasData {
    memset(_bias.contents + _outputChannel * 2, 0, (_outputChannelX4 - _outputChannel) * 2);
    memcpy(_bias.contents, biasData, _outputChannel * 2);
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
    /**set the function constant*/
    MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
    [constantValues setConstantValue:&_inputChannel    type:MTLDataTypeInt atIndex:0];
    [constantValues setConstantValue:&_outputChannel   type:MTLDataTypeInt atIndex:1];
    [constantValues setConstantValue:&_inputChannelX4  type:MTLDataTypeInt atIndex:2];
    [constantValues setConstantValue:&_outputChannelX4 type:MTLDataTypeInt atIndex:3];

    /**get the function*/
    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:_functionName constantValues:constantValues error:&error];

    if (!function) {
        NSLog(@"%@ init  function error:%@", _label, error);

        return;
    }

    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];

    if (!_computePipelineState) {
        NSLog(@"%@ init MTLComputePipelineState error with:%@", _label, error);
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
    /**init MTLComputeCommandEncoder*/
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    [encoder setBuffer:input    offset:0 atIndex:0];
    [encoder setBuffer:_weights offset:0 atIndex:1];

    if (_haveBias) {
        [encoder setBuffer:_bias    offset:0 atIndex:2];
        [encoder setBuffer:output   offset:0 atIndex:3];
    } else {
        [encoder setBuffer:output   offset:0 atIndex:2];
    }

    NSUInteger executeWidth = _computePipelineState.threadExecutionWidth;

    MTLSize threadsPerThreadgroup = MTLSizeMake(executeWidth, 1, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_outputChannel + 4 * executeWidth - 1) / (4 * executeWidth), 1, 1);

    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

    [encoder endEncoding];
}

/**
 * return the bytes of output
 */
- (int)getOutputBytes {
    return _outputChannelX4 * 2;
}

- (int)getInputBytes {
    return _inputChannelX4 * 2;
}

@end




















