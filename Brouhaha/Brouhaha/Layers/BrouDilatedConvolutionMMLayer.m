/**
 * BrouDilatedConvolutionMMLayer.h
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/2.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouConvolutionMMLayer.h"
#import "BrouDilatedConvolutionMMLayer.h"

@implementation BrouDilatedConvolutionMMLayer

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
                              strideY:(int)strideY
                             dilatedX:(int)dilatedX
                             dilatedY:(int)dilatedY {
    self = [super initWithLabel:@"BrouDilatedConvolutionMMLayer"];

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
                              strideY:strideY
                             dilatedX:dilatedX
                             dilatedY:dilatedY];

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
                              strideY:(int)strideY
                             dilatedX:(int)dilatedX
                             dilatedY:(int)dilatedY {
    self = [super initWithLabel:@"BrouDilatedConvolutionMMLayer"];

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
                              strideY:strideY
                             dilatedX:dilatedX
                             dilatedY:dilatedY];

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
                            strideY:(int)strideY
                           dilatedX:(int)dilatedX
                           dilatedY:(int)dilatedY {
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
    
    _dilatedX = dilatedX;
    _dilatedY = dilatedY;

    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;

    _M = _outputHeight * _outputWidth;
    _K = _kernelHeight * _kernelWidth * _inputChannel;
    _N = _outputChannel;

    _MX4 = (_M + 3) / 4 * 4;
    _NX4 = (_N + 3) / 4 * 4;

    _convertInputFunctionName = @"brouConvertInput2MatrixDilated";
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
    [constantValues setConstantValue:&_dilatedX        type:MTLDataTypeInt atIndex:12];
    [constantValues setConstantValue:&_dilatedY        type:MTLDataTypeInt atIndex:13];
    [constantValues setConstantValue:&_inputChannelX4  type:MTLDataTypeInt atIndex:14];
    [constantValues setConstantValue:&_outputChannelX4 type:MTLDataTypeInt atIndex:15];

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

@end















