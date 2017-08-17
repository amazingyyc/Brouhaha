/**
 * BrouDilatedConvolutionLayer.h
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/2.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouDilatedConvolutionLayer.h"

@interface BrouDilatedConvolutionLayer() {
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
}

@end

@implementation BrouDilatedConvolutionLayer

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
    self = [super initWithLabel:@"BrouDilatedConvolutionLayer"];
    
    if (!self) {
        return self;
    }
    
    if (biasData) {
        _haveBias = true;
        
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
        
        [self copyBufferWithKernel:float16Kernel];
        [self copyBufferWithBias:float16Bias];
        
        free(float16Kernel);
        free(float16Bias);
    } else {
        _haveBias = false;
        
        _haveBias = NO;
        
        /**malloc memory to store kernel*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * outputChannel * kernelHeight * kernelWidth * inputChannel);
        
        /**convert float32 to float16*/
        convertFloat32ToFloat16((uint32_t*)kernelData,
                                (uint16_t*)float16Kernel,
                                outputChannel * kernelHeight * kernelWidth * inputChannel);
        
        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _bias = [device newBufferWithLength:2
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        [self copyBufferWithKernel:float16Kernel];
        
        free(float16Kernel);
    }
    
    _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = _inputHeight;
    inputShapeRef->dim1 = _inputWidth;
    inputShapeRef->dim2 = _inputChannelX4;
    
    _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = _outputHeight;
    outputShapeRef->dim1 = _outputWidth;
    outputShapeRef->dim2 = _outputChannelX4;
    
    _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                            options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    
    convolutionShapeRef->padTop  = _padTop;
    convolutionShapeRef->padLeft = _padLeft;
    convolutionShapeRef->strideY = _strideY;
    convolutionShapeRef->strideX = _strideX;
    convolutionShapeRef->insertY = 0;
    convolutionShapeRef->insertX = 0;
    convolutionShapeRef->dilatedY = _dilatedY;
    convolutionShapeRef->dilatedX = _dilatedX;
    convolutionShapeRef->haveBias = _haveBias;
    
    _functionName = @"brouDilatedConvolution";
    
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
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
   
    /**get the function*/
    NSError *error = nil;
    
    id<MTLFunction> function = [library newFunctionWithName:_functionName];
    
    NSAssert(function, @"init function error!");

    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    
    NSAssert(_computePipelineState, @"init _computePipelineState error!");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input                offset:0 atIndex:0];
    [encoder setBuffer:_kernel              offset:0 atIndex:1];
    [encoder setBuffer:_bias                offset:0 atIndex:2];
    [encoder setBuffer:output               offset:0 atIndex:3];
    [encoder setBuffer:_inputShape          offset:0 atIndex:4];
    [encoder setBuffer:_outputShape         offset:0 atIndex:5];
    [encoder setBuffer:_convolutionShape    offset:0 atIndex:6];
    
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

@end











