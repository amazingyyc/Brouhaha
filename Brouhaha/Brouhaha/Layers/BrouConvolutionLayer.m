/**
 * BrouLayer.h
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the convolution layer
 */

#import <Metal/Metal.h>

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouLayer.h"
#import "BrouConvolutionLayer.h"

@implementation BrouConvolutionLayer

/**
 init the convolution compute kernel

 @param device metal device
 @param kernelData the convolution kernel must be flaot32
 @param biasData the bias must the float32
 @param inputHeight the input's height
 @param inputWidth the input's width
 @param inputChannel the input chcnnel
 @param outputHeight the output height
 @param outputWidth output width
 @param outputChannel output channel
 @param kernelHeight the kernel height
 @param kernelWidth the kernel width
 @param padLeft the pad on x axis
 @param padTop the pad on y axis
 @param strideX the stride of x axis
 @param strideY the stride of y axis
 @return the convolution operator
 */
- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)kernelData
                                 bias:(void*)biasData
                          inputHeight:(int)inputHeight
                           inputWidth:(int)inputWidth
                        intputChannel:(int)inputChannel
                         outputHeight:(int)outputHeight
                          outputWidth:(int)outputWidth
                        outputChannel:(int)outputChannel
                         kernelHeight:(int)kernelHeight
                          kernelWidth:(int)kernelWidth
                              padLeft:(int)padLeft
                               padTop:(int)padTop
                              strideX:(int)strideX
                              strideY:(int)strideY {
    self = [super initWithLabel:@"BrouConvolutionLayer"];

    if (!self) {
        return self;
    }

    [self configParametersInputHeight:inputHeight
                           inputWidth:inputWidth
                        intputChannel:inputChannel
                         outputHeight:outputHeight
                          outputWidth:outputWidth
                        outputChannel:outputChannel
                         kernelHeight:kernelHeight
                          kernelWidth:kernelWidth
                              padLeft:padLeft
                               padTop:padTop
                              strideX:strideX
                              strideY:strideY];

    /**if it has a bias*/
    if (biasData) {
        _haveBias = YES;

        /**malloc memory to store kernel and bias*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * outputChannel * kernelHeight * kernelWidth * inputChannel);
        void* float16Bias   = (void*)malloc(sizeof(uint16_t) * outputChannel);

        /**convert float32 to float16*/
        convertFloat32ToFloat16Two((uint32_t *)kernelData,
                                   (uint16_t *)float16Kernel,
                                   outputChannel * kernelHeight * kernelWidth * inputChannel,
                                   (uint32_t *)biasData,
                                   (uint16_t *)float16Bias,
                                   outputChannel);

        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        _bias = [device newBufferWithLength:2 * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        /**copy data to MTLBuffer*/
        [self copyBufferWithKernel:float16Kernel];
        [self copyBufferWithBias:float16Bias];

        free(float16Kernel);
        free(float16Bias);
    } else {
        _haveBias = NO;

        /**malloc memory to store kernel*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * outputChannel * kernelHeight * kernelWidth * inputChannel);

        /**convert float32 to float16*/
        convertFloat32ToFloat16((uint32_t*)kernelData,
                                (uint16_t*)float16Kernel,
                                outputChannel * kernelHeight * kernelWidth * inputChannel);

        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        _bias = nil;

        [self copyBufferWithKernel:float16Kernel];

        free(float16Kernel);
    }

    [self buildComputePipelinesStateWithDevice:device library:library];

    return self;
}

/**
 * the kernel and bias is float16
 */
- (instancetype)initWithFloat16Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)float16Kernel
                                 bias:(void*)float16Bias
                          inputHeight:(int)inputHeight
                           inputWidth:(int)inputWidth
                        intputChannel:(int)inputChannel
                         outputHeight:(int)outputHeight
                          outputWidth:(int)outputWidth
                        outputChannel:(int)outputChannel
                         kernelHeight:(int)kernelHeight
                          kernelWidth:(int)kernelWidth
                              padLeft:(int)padLeft
                               padTop:(int)padTop
                              strideX:(int)strideX
                              strideY:(int)strideY {
    self = [super initWithLabel:@"BrouConvolutionLayer"];

    if (!self) {
        return self;
    }

    [self configParametersInputHeight:inputHeight
                           inputWidth:inputWidth
                        intputChannel:inputChannel
                         outputHeight:outputHeight
                          outputWidth:outputWidth
                        outputChannel:outputChannel
                         kernelHeight:kernelHeight
                          kernelWidth:kernelWidth
                              padLeft:padLeft
                               padTop:padTop
                              strideX:strideX
                              strideY:strideY];

    if (float16Bias) {
        _haveBias = YES;

        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        _bias = [device newBufferWithLength:2 * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        /**copy data to MTLBuffer*/
        [self copyBufferWithKernel:float16Kernel];
        [self copyBufferWithBias:float16Bias];
    } else {
        _haveBias = NO;

        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

        _bias = nil;

        [self copyBufferWithKernel:float16Kernel];
    }

    [self buildComputePipelinesStateWithDevice:device library:library];

    return self;
}

- (void)configParametersInputHeight:(int)inputHeight
                         inputWidth:(int)inputWidth
                      intputChannel:(int)inputChannel
                       outputHeight:(int)outputHeight
                        outputWidth:(int)outputWidth
                      outputChannel:(int)outputChannel
                       kernelHeight:(int)kernelHeight
                        kernelWidth:(int)kernelWidth
                            padLeft:(int)padLeft
                             padTop:(int)padTop
                            strideX:(int)strideX
                            strideY:(int)strideY {
    /**ths input dimension is (inputHeight, inputWidth, intputChannel)*/
    _inputHeight  = inputHeight;
    _inputWidth   = inputWidth;
    _inputChannel = inputChannel;

    /**the output diemsnion is (outputHeight, outputWidth, outputChannel)*/
    _outputHeight  = outputHeight;
    _outputWidth   = outputWidth;
    _outputChannel = outputChannel;

    /**the kernel dimesnion is (outputChannel, kernelHeight, kernelWidth, inputChannel)*/
    _kernelHeight = kernelHeight;
    _kernelWidth  = kernelWidth;

    /**the pad*/
    _padLeft = padLeft;
    _padTop  = padTop;

    /**stride of kernel*/
    _strideX = strideX;
    _strideY = strideY;

    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;
    
    /**function name*/
    _functionName = @"brouConvolution";
    
    if (3 == _kernelHeight && 3 == _kernelWidth) {
        if (1 == _strideY && 1 == _strideX) {
            _functionName = @"brouConvolutionKernel3X3Stride1X1";
        } else if (2 == _strideY && 2 == _strideX) {
            _functionName = @"brouConvolutionKernel3X3Stride2X2";
        } else {
            _functionName = @"brouConvolutionKernel3X3";
        }
    }
}

/**
 * copy float16 kernel and bias to MTLBuffer
 */
- (void)copyBufferWithKernel:(void*)float16Kernel {
    /**set zero*/
    memset(_kernel.contents, 0, _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4 * 2);
    
    for (int i = 0; i < _outputChannel; ++i) {
        void *kernelBuffer  = _kernel.contents + i * _kernelHeight * _kernelWidth * _inputChannelX4 * 2;
        void *float16Buffer = float16Kernel + i * _kernelHeight * _kernelWidth * _inputChannel * 2;
        
        for (int j = 0; j < _kernelHeight * _kernelWidth; ++j) {
            memcpy(kernelBuffer, float16Buffer, _inputChannel * 2);
            
            kernelBuffer  += _inputChannelX4 * 2;
            float16Buffer += _inputChannel * 2;
        }
    }
}

- (void)copyBufferWithBias:(void*)float16Bias {
    memset(_bias.contents, 0, _outputChannelX4 * 2);
    memcpy(_bias.contents, float16Bias, _outputChannel * 2);
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
    [constantValues setConstantValue:&_inputChannel    type:MTLDataTypeInt atIndex:2];
    [constantValues setConstantValue:&_outputHeight    type:MTLDataTypeInt atIndex:3];
    [constantValues setConstantValue:&_outputWidth     type:MTLDataTypeInt atIndex:4];
    [constantValues setConstantValue:&_outputChannel   type:MTLDataTypeInt atIndex:5];
    [constantValues setConstantValue:&_kernelHeight    type:MTLDataTypeInt atIndex:6];
    [constantValues setConstantValue:&_kernelWidth     type:MTLDataTypeInt atIndex:7];
    [constantValues setConstantValue:&_padLeft         type:MTLDataTypeInt atIndex:8];
    [constantValues setConstantValue:&_padTop          type:MTLDataTypeInt atIndex:9];
    [constantValues setConstantValue:&_strideX         type:MTLDataTypeInt atIndex:10];
    [constantValues setConstantValue:&_strideY         type:MTLDataTypeInt atIndex:11];
    [constantValues setConstantValue:&_inputChannelX4  type:MTLDataTypeInt atIndex:12];
    [constantValues setConstantValue:&_outputChannelX4 type:MTLDataTypeInt atIndex:13];
    [constantValues setConstantValue:&_haveBias        type:MTLDataTypeBool atIndex:14];

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
 
 the input's dimension should be (inputHeight, intputWidth, intputChannelX4)
 the output's dimension should be (outputHeight, outputWidth, outputChannelX4)
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];

    [encoder setBuffer:input   offset:0 atIndex:0];
    [encoder setBuffer:_kernel offset:0 atIndex:1];
    [encoder setBuffer:_bias   offset:0 atIndex:2];
    [encoder setBuffer:output  offset:0 atIndex:3];

    /**
     * every thread will handle 4X4X4 output
     */
    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_outputWidth  + 31) / 32,
                                                (_outputHeight + 15) / 16,
                                                _outputChannelX4 / 4);

    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

/**
 * return the bytes of output
 */
- (int)getOutputBytes {
    return _outputHeight * _outputWidth * _outputChannelX4 * 2;
}

- (int)getInputBytes {
    return _inputHeight * _inputWidth * _inputChannelX4 * 2;
}

@end









