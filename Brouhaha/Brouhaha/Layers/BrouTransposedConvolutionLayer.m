/**
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/30.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouTransposedConvolutionLayer.h"

@implementation BrouTransposedConvolutionLayer

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)kernelData
                                 bias:(void*)biasData
                    originInputHeight:(int)originInputHeight
                     originInputWidth:(int)originInputWidth
                   originInputChannel:(int)originInputChannel
                   originOutputHeight:(int)originOutputHeight
                    originOutputWidth:(int)originOutputWidth
                  originOutputChannel:(int)originOutputChannel
                   originKernelHeight:(int)originKernelHeight
                    originKernelWidth:(int)originKernelWidth
                        originPadLeft:(int)originPadLeft
                         originPadTop:(int)originPadTop
                        originStrideX:(int)originStrideX
                        originStrideY:(int)originStrideY
                       outputAddRight:(int)outputAddRight
                      outputAddBottom:(int)outputAddBottom {
    self = [super initWithLabel:@"BrouTransposedConvolutionLayer"];
    
    if (!self) {
        return self;
    }
    
    [self configParametersOriginInputHeight:originInputHeight
                           originInputWidth:originInputWidth
                         originInputChannel:originInputChannel
                         originOutputHeight:originOutputHeight
                          originOutputWidth:originOutputWidth
                        originOutputChannel:originOutputChannel
                         originKernelHeight:originKernelHeight
                          originKernelWidth:originKernelWidth
                              originPadLeft:originPadLeft
                               originPadTop:originPadTop
                              originStrideX:originStrideX
                              originStrideY:originStrideY
                             outputAddRight:outputAddRight
                            outputAddBottom:outputAddBottom];
    if (biasData) {
        _haveBias = true;
        
        /**malloc memory to store kernel and bias*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
        void* float16Bias   = (void*)malloc(sizeof(uint16_t) * _outputChannel);
        
        /**convert float32 to float16*/
        convertFloat32ToFloat16Two((uint32_t *)kernelData,
                                   (uint16_t *)float16Kernel,
                                   _outputChannel * _kernelHeight * _kernelWidth * _inputChannel,
                                   (uint32_t *)biasData,
                                   (uint16_t *)float16Bias,
                                   _outputChannel);
        
        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _bias = [device newBufferWithLength:2 * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        [self copyBufferWithKernel:float16Kernel];
        [self copyBufferWithBias:float16Bias];
    } else {
        _haveBias = false;
        
        /**malloc memory to store kernel and bias*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
        
        /**convert float32 to float16*/
        convertFloat32ToFloat16((uint32_t*)kernelData,
                                (uint16_t*)float16Kernel,
                                _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
        
        _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        /**?? set the bias is nil or zero length will get a error, do not know why*/
        _bias = [device newBufferWithLength:2
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        [self copyBufferWithKernel:float16Kernel];
    }
    
    _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                            options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    TensorShape *tensorShapeRef = (TensorShape*)_inputShape.contents;
    tensorShapeRef->dim0 = _inputHeight;
    tensorShapeRef->dim1 = _inputWidth;
    tensorShapeRef->dim2 = _inputChannelX4;
    
    tensorShapeRef = (TensorShape*)_outputShape.contents;
    tensorShapeRef->dim0 = _outputHeight;
    tensorShapeRef->dim1 = _outputWidth;
    tensorShapeRef->dim2 = _outputChannelX4;

    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    
    convolutionShapeRef->padTop  = _padTop;
    convolutionShapeRef->padLeft = _padLeft;
    convolutionShapeRef->strideY = _strideY;
    convolutionShapeRef->strideX = _strideX;
    convolutionShapeRef->insertY = _insertY;
    convolutionShapeRef->insertX = _insertX;
    convolutionShapeRef->haveBias = _haveBias;
    
    _functionName = @"brouTransposedConvolution";
    
    [self buildComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configParametersOriginInputHeight:(int)originInputHeight
                         originInputWidth:(int)originInputWidth
                       originInputChannel:(int)originInputChannel
                       originOutputHeight:(int)originOutputHeight
                        originOutputWidth:(int)originOutputWidth
                      originOutputChannel:(int)originOutputChannel
                       originKernelHeight:(int)originKernelHeight
                        originKernelWidth:(int)originKernelWidth
                            originPadLeft:(int)originPadLeft
                             originPadTop:(int)originPadTop
                            originStrideX:(int)originStrideX
                            originStrideY:(int)originStrideY
                           outputAddRight:(int)outputAddRight
                          outputAddBottom:(int)outputAddBottom {
    _originInputHeight  = originInputHeight;
    _originInputWidth   = originInputWidth;
    _originInputChannel = originInputChannel;
    
    _originOutputeHeight  = originOutputHeight;
    _originOutputWidth    = originOutputWidth;
    _originOutputChannnel = originOutputChannel;
    
    _originKernelHeight = originKernelHeight;
    _originKernelWidth  = originKernelWidth;
    
    _originPadLeft = originPadLeft;
    _originPadTop  = originPadTop;
    
    _originStrideX = originStrideX;
    _originStrideY = originStrideY;
    
    _outputAddRight  = outputAddRight;
    _outputAddBottom = outputAddBottom;
    
    /**calculate the transposed convolution parammers*/
    _inputHeight  = _originOutputeHeight;
    _inputWidth   = _originOutputWidth;
    _inputChannel = _originOutputChannnel;
    
    _outputHeight  = _originInputHeight;
    _outputWidth   = _originInputWidth;
    _outputChannel = _originInputChannel;
    
    _kernelHeight = _originKernelHeight;
    _kernelWidth  = _originKernelWidth;
    
    _padLeft =  _originKernelWidth  - _originPadLeft - 1;
    _padTop  =  _originKernelHeight - _originPadTop  - 1;
    
    _strideX = 1;
    _strideY = 1;
    
    _insertX = _originStrideX - 1;
    _insertY = _originStrideY - 1;
    
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
    
    [encoder setBuffer:input             offset:0 atIndex:0];
    [encoder setBuffer:_kernel           offset:0 atIndex:1];
    [encoder setBuffer:_bias             offset:0 atIndex:2];
    [encoder setBuffer:output            offset:0 atIndex:3];
    [encoder setBuffer:_inputShape       offset:0 atIndex:4];
    [encoder setBuffer:_outputShape      offset:0 atIndex:5];
    [encoder setBuffer:_convolutionShape offset:0 atIndex:6];
        
    /**
     * every thread will handle 4X4X4 output
     */
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((_outputWidth  + 31) / 32,
                                (_outputHeight + 15) / 16,
                                _outputChannelX4 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

























