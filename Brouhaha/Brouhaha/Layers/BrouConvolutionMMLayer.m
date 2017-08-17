/**
 * Created by yanyuanchi on 2017/5/30.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 * 
 * use the matrix multiply to calculate the convolution
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouConvolutionMMLayer.h"


@implementation BrouConvolutionMMLayer

/**
 init the convolution compute kernel

 @param device metal device
 @param kernelData the convolution kernel must be flaot32
 @param biasData the bias must the float32
 @param inputHeight the input's height
 @param inputWidth the input's width
 @param inputChannel the input channel
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

    self = [super initWithLabel:@"BrouConvolutionMMLayer"];

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

        [self configBufferWithDevice:device kernel:float16Kernel];
        [self configBufferWithDevice:device bias:float16Bias];

        free(float16Kernel);
        free(float16Bias);

        _functionName = @"brouMatrixMultiply";
    } else {
        _haveBias = NO;

        /**malloc memory to store kernel and bias*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * outputChannel * kernelHeight * kernelWidth * inputChannel);

        convertFloat32ToFloat16((uint32_t*)kernelData,
                                (uint16_t*)float16Kernel,
                                outputChannel * kernelHeight * kernelWidth * inputChannel);

        [self configBufferWithDevice:device kernel:float16Kernel];

        free(float16Kernel);

        _functionName = @"brouMatrixMultiplyWithoutBias";
    }

    /**config buffer*/
    [self configFunctionWithDevice:device library:library];

    return self;
}

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

    self = [super initWithLabel:@"BrouConvolutionMMLayer"];

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

        [self configBufferWithDevice:device kernel:float16Kernel];
        [self configBufferWithDevice:device bias:float16Bias];

        _functionName = @"brouMatrixMultiply";
    } else {
        _haveBias = NO;

        [self configBufferWithDevice:device kernel:float16Kernel];

        _functionName = @"brouMatrixMultiplyWithoutBias";
    }

    /**config buffer*/
    [self configFunctionWithDevice:device library:library];

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
    [super configParametersInputHeight:inputHeight
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

    _M = _outputHeight * _outputWidth;
    _K = _kernelHeight * _kernelWidth * _inputChannel;
    _N = _outputChannel;

    _MX4 = (_M + 3) / 4 * 4;
    _NX4 = (_N + 3) / 4 * 4;

    _convertInputFunctionName = @"brouConvertInput2Matrix";
}

- (void)configBufferWithDevice:(id<MTLDevice>)device kernel:(void*)float16Kernel {
    /**strore the 2d input data*/
    _inputMatrix = [device newBufferWithLength:2 * _K * _MX4
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    /**the 4d kernel will convert to a matrix*/
    _kernel = [device newBufferWithLength:2 * _K * _NX4
                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    brouTransposeMatrix_uint16_t(float16Kernel,
                                 _outputChannel, _kernelHeight*_kernelWidth*_inputChannel,
                                 _kernel.contents,
                                 _kernelHeight*_kernelWidth*_inputChannel, _outputChannelX4);
}

- (void)configBufferWithDevice:(id<MTLDevice>)device bias:(void*)float16Bias {
    /**store the bias*/
    _bias = [device newBufferWithLength:2 * _outputChannelX4
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    /**copy bias data*/
    memset(_bias.contents + _outputChannel * 2, 0, (_outputChannelX4 - _outputChannel) * 2);
    memcpy(_bias.contents, float16Bias, _outputChannel * 2);
}

/**
 * config the function
 */
- (void)configFunctionWithDevice:(id<MTLDevice>)device library:(id<MTLLibrary>)library {
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

    /**get the function*/
    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:_convertInputFunctionName constantValues:constantValues error:&error];

    if (!function) {
        NSLog(@"init  function error");

        return;
    }

    _convertInputComputePipelineState = [device newComputePipelineStateWithFunction:function error:&error];

    if (!_convertInputComputePipelineState) {
        NSLog(@"init MTLComputePipelineState error");
    }

    /**set matrix multiply params*/
    [constantValues reset];

    [constantValues setConstantValue:&_M   type:MTLDataTypeInt atIndex:0];
    [constantValues setConstantValue:&_K   type:MTLDataTypeInt atIndex:1];
    [constantValues setConstantValue:&_N   type:MTLDataTypeInt atIndex:2];
    [constantValues setConstantValue:&_MX4 type:MTLDataTypeInt atIndex:3];
    [constantValues setConstantValue:&_NX4 type:MTLDataTypeInt atIndex:4];

    function = [library newFunctionWithName:_functionName
                             constantValues:constantValues
                                      error:&error];

    if (!function) {
        NSLog(@"init  convert input function error");
        return;
    }

    _computePipelineState = [device newComputePipelineStateWithFunction:function
                                                                  error:&error];

    if (!_computePipelineState) {
        NSLog(@"init convertInputComputePipelineState error");
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
    /**config convert input to matrix encoder*/
    id<MTLComputeCommandEncoder> convertInputEncode = [commandBuffer computeCommandEncoder];
    [convertInputEncode setComputePipelineState:_convertInputComputePipelineState];
    [convertInputEncode setBuffer:input        offset:0 atIndex:0];
    [convertInputEncode setBuffer:_inputMatrix offset:0 atIndex:1];

    MTLSize threadsPerThreadgroup = MTLSizeMake(32, 1, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_MX4  + 127) / 128,
                                                1,
                                                1);

    [convertInputEncode dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [convertInputEncode endEncoding];

    /**config the matrix multiply enode*/
    id<MTLComputeCommandEncoder> multiplyEncode = [commandBuffer computeCommandEncoder];
    [multiplyEncode setComputePipelineState:_computePipelineState];
    [multiplyEncode setBuffer:_inputMatrix offset:0 atIndex:0];
    [multiplyEncode setBuffer:_kernel      offset:0 atIndex:1];

    if (_haveBias) {
        [multiplyEncode setBuffer:_bias  offset:0 atIndex:2];
        [multiplyEncode setBuffer:output offset:0 atIndex:3];
    } else {
        [multiplyEncode setBuffer:output offset:0 atIndex:2];
    }

    threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    threadgroupsPerGrid   = MTLSizeMake((_NX4 + 31) / 32,
                                        (_MX4 + 15) / 16,
                                        1);

    [multiplyEncode dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [multiplyEncode endEncoding];
}

/**
 * return the bytes of output
 * the output must be bigger than ([_outputHeight * _outputWidth]X4, _outputChannelX4)
 */
- (int)getOutputBytes {
    return _MX4 * _NX4 * 2;
}

- (int)getInputBytes {
    return _inputHeight * _inputWidth * _inputChannelX4 * 2;
}

@end



























