/**
 * BrouTransposedConvolutionMMLayer.m
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/2.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */
#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouTransposedConvolutionMMLayer.h"

@interface BrouTransposedConvolutionMMLayer() {
    /**
     * _inputMatrixRow = inputChannelx4
     * _inputMatrixCol = [inputHeight * inputWidth]X4
     */
    int _inputMatrixRow;
    int _inputMatrixCol;
    
    /**
     * _mediateMatrixRow = _outputChannelX4 * kernelHeight * kernelWidth
     * _mediateMatrixCol = _inputMatrixCol = [inputHeight * inputWidth]X4
     */
    int _mediateMatrixRow;
    int _mediateMatrixCol;
    
    /**
     * _kernelMatrixRow = inputChannelx4
     * _kernelMatrixCol = _outputChannelX4 * kernelHeight * kernelWidth
     */
    int _kernelMatrixRow;
    int _kernelMatrixCol;
    
    id<MTLBuffer> _input2MatrixShape;
    id<MTLBuffer> _matrixMultipyShape;
    id<MTLBuffer> _matrix2OutputShape;
    
    id<MTLBuffer> _inputMatrix;
    id<MTLBuffer> _mediateMatrix;
    
    NSString *_input2MatrixFunctionName;
    NSString *_matrixMultiplyFunctionName;
    NSString *_matrix2OutputFunctionName;
    
    id<MTLComputePipelineState> _input2MatrixComputePipelineState;
    id<MTLComputePipelineState> _matrixMultiplyComputePipelineState;
    id<MTLComputePipelineState> _matrix2OutputComputePipelineState;
}

@end

@implementation BrouTransposedConvolutionMMLayer

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
    self = [super initWithLabel:@"BrouTransposedConvolutionMMLayer"];

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
        convertFloat32ToFloat16Two(kernelData,
                                   float16Kernel,
                                   _outputChannel * _kernelHeight * _kernelWidth * _inputChannel,
                                   biasData,
                                   float16Bias,
                                   _outputChannel);
        
        [self configBufferWithDevice:device kernel:float16Kernel];
        [self configBufferWithDevice:device kernel:float16Bias];
        
        free(float16Kernel);
        free(float16Bias);
    } else {
        _haveBias = false;
        
        /**malloc memory to store kernel and bias*/
        void* float16Kernel = (void*)malloc(sizeof(uint16_t) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
        
        convertFloat32ToFloat16(kernelData, float16Kernel, _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
        
        [self configBufferWithDevice:device kernel:float16Kernel];
        
        _bias = [device newBufferWithLength:2 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        free(float16Kernel);
    }
    
    _input2MatrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                             options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    TensorShape *input2MatrixShapeRef = (TensorShape*)_input2MatrixShape.contents;
    input2MatrixShapeRef->dim0 = _inputHeight * _inputWidth;
    input2MatrixShapeRef->dim1 = _inputMatrixRow;
    input2MatrixShapeRef->dim2 = _inputMatrixCol;
    
    _matrixMultipyShape = [device newBufferWithLength:sizeof(TensorShape)
                                              options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    TensorShape *matrixMultipyShapeRef = (TensorShape*)_matrixMultipyShape.contents;
    matrixMultipyShapeRef->dim0  = _mediateMatrixRow;
    matrixMultipyShapeRef->dim1  = _kernelMatrixRow;
    matrixMultipyShapeRef->dim2  = _mediateMatrixCol;
    
    _matrix2OutputShape = [device newBufferWithLength:sizeof(TensorShape)
                                              options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    TensorShape *matrix2OutputShapeRef = (TensorShape*)_matrix2OutputShape.contents;
    matrix2OutputShapeRef->dim0 = _mediateMatrixRow;
    matrix2OutputShapeRef->dim1 = _mediateMatrixCol;
    
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
    convolutionShapeRef->insertY = _insertY;
    convolutionShapeRef->insertX = _insertX;
    convolutionShapeRef->haveBias = _haveBias;
    
    _input2MatrixFunctionName   = @"brouConvertInput2MatrixTransposed";
    _matrixMultiplyFunctionName = @"brouMatrixMultiplyWithShape";
    _matrix2OutputFunctionName  = @"brouConvertMatrix2OutputTransposed";
    
    _inputMatrix = [device newBufferWithLength:2 * _inputMatrixRow * _inputMatrixCol
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _mediateMatrix = [device newBufferWithLength:2 * _mediateMatrixRow * _mediateMatrixCol
                                         options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    /**get metal function*/
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
    [super configParametersOriginInputHeight:originInputHeight
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
    
    _inputMatrixRow = _inputChannelX4;
    _inputMatrixCol = (_inputHeight * _inputWidth + 3) / 4 * 4;
    
    _kernelMatrixRow = _inputChannelX4;
    _kernelMatrixCol = _outputChannelX4 * _kernelHeight * _kernelWidth;
    
    _mediateMatrixRow = _outputChannelX4 * _kernelHeight * _kernelWidth;
    _mediateMatrixCol = _inputMatrixCol;
}

/**
 * config the kernel buffer
 */
- (void)configBufferWithDevice:(id<MTLDevice>)device kernel:(void*)float16Kernel {
    _kernel = [device newBufferWithLength:2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    memset(_kernel.contents, 0, 2 * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4);
    
    /**transpose the kernel*/
    brouTransposeMatrix_uint16_t(float16Kernel,
                                 _outputChannel * _kernelHeight * _kernelWidth,
                                 _inputChannel,
                                 _kernel.contents,
                                 _kernelMatrixRow,
                                 _kernelMatrixCol);
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
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {
    
    /**get the function*/
    NSError *error = nil;
    
    /**input 2 matrix*/
    id<MTLFunction> input2MatrixFunction = [library newFunctionWithName:_input2MatrixFunctionName];
    
    NSAssert(input2MatrixFunction, @"init input2Matrix function error!");

    _input2MatrixComputePipelineState = [device newComputePipelineStateWithFunction:input2MatrixFunction
                                                                              error:&error];
    
    NSAssert(_input2MatrixComputePipelineState, @"input2Matrix function error!");

    /**matrix multiply*/
    id<MTLFunction> matrixMultiplyFunction = [library newFunctionWithName:_matrixMultiplyFunctionName];

    NSAssert(matrixMultiplyFunction, @"init matrix multiply function error!");
    
    _matrixMultiplyComputePipelineState = [device newComputePipelineStateWithFunction:matrixMultiplyFunction
                                                                                error:&error];
    
    NSAssert(_matrixMultiplyComputePipelineState, @"init matrix multiply function error!");
    
    /**matrix to output*/
    id<MTLFunction> matrix2OutputFunction = [library newFunctionWithName:_matrix2OutputFunctionName];
    
    NSAssert(matrix2OutputFunction, @"init matrix 2 output function error!");
    
    _matrix2OutputComputePipelineState = [device newComputePipelineStateWithFunction:matrix2OutputFunction
                                                                               error:&error];
    
    NSAssert(_matrix2OutputComputePipelineState, @"init matrix 2 output function error!");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> input2MatrixEncoder = [commandBuffer computeCommandEncoder];
    [input2MatrixEncoder setComputePipelineState:_input2MatrixComputePipelineState];
    [input2MatrixEncoder setBuffer:input              offset:0 atIndex:0];
    [input2MatrixEncoder setBuffer:_inputMatrix       offset:0 atIndex:1];
    [input2MatrixEncoder setBuffer:_input2MatrixShape offset:0 atIndex:2];
    
    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_inputMatrixCol + 31) / 32,
                                                (_inputMatrixRow + 15) / 16,
                                                1);
    
    [input2MatrixEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [input2MatrixEncoder endEncoding];
    
    id<MTLComputeCommandEncoder> matrixMultipyEncoder = [commandBuffer computeCommandEncoder];
    [matrixMultipyEncoder setComputePipelineState:_matrixMultiplyComputePipelineState];
    [matrixMultipyEncoder setBuffer:_kernel             offset:0 atIndex:0];
    [matrixMultipyEncoder setBuffer:_inputMatrix        offset:0 atIndex:1];
    [matrixMultipyEncoder setBuffer:_mediateMatrix      offset:0 atIndex:2];
    [matrixMultipyEncoder setBuffer:_matrixMultipyShape offset:0 atIndex:3];
    
    threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    threadgroupsPerGrid   = MTLSizeMake((_mediateMatrixCol + 31) / 32,
                                        (_mediateMatrixRow + 15) / 16,
                                        1);
    [matrixMultipyEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [matrixMultipyEncoder endEncoding];
    
    /**
     device half *matrix               [[buffer(0)]],
     device half *output               [[buffer(1)]],
     device half *bia                  [[buffer(2)]],
     constant TensorShape& matrixShape [[buffer(3)]],
     constant TensorShape& inputShape  [[buffer(4)]],
     constant TensorShape& outputShape [[buffer(5)]],
     constant ConvolutionShape& convolutionShape [[buffer(6)]],
     */
    id<MTLComputeCommandEncoder> matrix2OutputEncoder = [commandBuffer computeCommandEncoder];
    [matrix2OutputEncoder setComputePipelineState:_matrix2OutputComputePipelineState];
    [matrix2OutputEncoder setBuffer:_mediateMatrix         offset:0 atIndex:0];
    [matrix2OutputEncoder setBuffer:output                 offset:0 atIndex:1];
    [matrix2OutputEncoder setBuffer:_bias                  offset:0 atIndex:2];
    [matrix2OutputEncoder setBuffer:_matrix2OutputShape    offset:0 atIndex:3];
    [matrix2OutputEncoder setBuffer:_inputShape            offset:0 atIndex:4];
    [matrix2OutputEncoder setBuffer:_outputShape           offset:0 atIndex:5];
    [matrix2OutputEncoder setBuffer:_convolutionShape      offset:0 atIndex:6];
    
    threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    threadgroupsPerGrid   = MTLSizeMake((_outputWidth  + 31) / 32,
                                        (_outputHeight + 15) / 16,
                                        _outputChannel);
    
    [matrix2OutputEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [matrix2OutputEncoder endEncoding];
}

@end














