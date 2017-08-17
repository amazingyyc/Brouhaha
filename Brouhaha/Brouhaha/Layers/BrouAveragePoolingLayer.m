/**
 * BrouAveragePoolingLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the average pooling layer
 */

#import "BrouAveragePoolingLayer.h"

@interface BrouAveragePoolingLayer() {
    BOOL _includePad;
}

@end

@implementation BrouAveragePoolingLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   InputHeight:(int)inputHeight
                         inputWidth:(int)inputWidth
                       outputHeight:(int)outputHeight
                        outputWidth:(int)outputWidth
                            channel:(int)channel
                       kernelHeight:(int)kernelHeight
                        kernelWidth:(int)kerneWidth
                            padLeft:(int)padLeft
                             padTop:(int)padTop
                            strideX:(int)strideX
                            strideY:(int)strideY
                         includePad:(BOOL)includePad {
    self = [super initWithInputHeight:inputHeight
                           inputWidth:inputWidth
                         outputHeight:outputHeight
                          outputWidth:outputWidth
                              channel:channel
                         kernelHeight:kernelHeight
                          kernelWidth:kerneWidth
                              padLeft:padLeft
                               padTop:padTop
                              strideX:strideX
                              strideY:strideY];

    if (!self) {
        return self;
    }

    _label = @"BrouAveragePoolingLayer";

    /**if include the pad when calculate the average*/
    _includePad = includePad;

    if (_includePad) {
        _functionName = @"brouAveragePooling";
    } else {
        _functionName = @"brouAveragePoolingWithoutPad";
    }

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
    [constantValues setConstantValue:&_inputHeight     type:MTLDataTypeInt atIndex:0];
    [constantValues setConstantValue:&_inputWidth      type:MTLDataTypeInt atIndex:1];
    [constantValues setConstantValue:&_outputHeight    type:MTLDataTypeInt atIndex:2];
    [constantValues setConstantValue:&_outputWidth     type:MTLDataTypeInt atIndex:3];
    [constantValues setConstantValue:&_channel         type:MTLDataTypeInt atIndex:4];
    [constantValues setConstantValue:&_kernelHeight    type:MTLDataTypeInt atIndex:5];
    [constantValues setConstantValue:&_kernelWidth     type:MTLDataTypeInt atIndex:6];
    [constantValues setConstantValue:&_padLeft         type:MTLDataTypeInt atIndex:7];
    [constantValues setConstantValue:&_padTop          type:MTLDataTypeInt atIndex:8];
    [constantValues setConstantValue:&_strideX         type:MTLDataTypeInt atIndex:9];
    [constantValues setConstantValue:&_strideY         type:MTLDataTypeInt atIndex:10];
    [constantValues setConstantValue:&_channelX4       type:MTLDataTypeInt atIndex:11];

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

 the input's dimension should be (inputHeight, intputWidth, channelX4)
 the output's dimension should be (outputHeight, outputWidth, channelX4)
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];

    [encoder setBuffer:input  offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];

    /**
     * every thread will handle 4X4X4 output
     */
    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_outputWidth  + 7) / 8,
                                                (_outputHeight + 3) / 4,
                                                _channelX4 / 4);

    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

/**
 * return the bytes of output
 */
- (int)getOutputBytes {
    return _outputHeight * _outputWidth * _channelX4 * 2;
}

- (int)getInputBytes {
    return _inputHeight * _inputWidth * _channelX4 * 2;
}

@end

















