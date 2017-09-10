#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(TransposedConvolutionMMLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   floatKernel:(void*)floatKernel
                     floatBias:(void*)floatBias
            originInputChannel:(int)originInputChannel
           originOutputChannel:(int)originOutputChannel
            originKernelHeight:(int)originKernelHeight
             originKernelWidth:(int)originKernelWidth
                  originPadTop:(int)originPadTop
                 originPadLeft:(int)originPadLeft
                 originStrideY:(int)originStrideY
                 originStrideX:(int)originStrideX {
    self = [super initWithName:@BROU_OBJECT_NAME(TransposedConvolutionMMLayer)];
    
    if (!self) {
        return self;
    }
    
    [self configParamsWithOriginInputChannel:originInputChannel
                         originOutputChannel:originOutputChannel
                          originKernelHeight:originKernelHeight
                           originKernelWidth:originKernelWidth
                                originPadTop:originPadTop
                               originPadLeft:originPadLeft
                               originStrideY:originStrideY
                               originStrideX:originStrideX];
    
    [self configBufferWithDevice:device floatKernel:floatKernel];
    [self configBufferWithDevice:device floatBias:floatBias];
    [self configShapeWithDevice:device];
    
    _input2MatrixFunctionName   = @BROU_METAL(TransposedConvolutionInput2Matrix);
    _matrixMultiplyFunctionName = @BROU_METAL(MatrixMultiply);
    _matrix2OutputFunctionName  = @BROU_METAL(TransposedConvolutionMatrix2Output);
    
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configParamsWithOriginInputChannel:(int)originInputChannel
                       originOutputChannel:(int)originOutputChannel
                        originKernelHeight:(int)originKernelHeight
                         originKernelWidth:(int)originKernelWidth
                              originPadTop:(int)originPadTop
                             originPadLeft:(int)originPadLeft
                             originStrideY:(int)originStrideY
                             originStrideX:(int)originStrideX {
    [super configParamsWithOriginInputChannel:originInputChannel
                          originOutputChannel:originOutputChannel
                           originKernelHeight:originKernelHeight
                            originKernelWidth:originKernelWidth
                                 originPadTop:originPadTop
                                originPadLeft:originPadLeft
                                originStrideY:originStrideY
                                originStrideX:originStrideX];
    
    _inputMatrixRow = _inputChannelX4;
    
    _mediateMatrixRow = _outputChannelX4 * _kernelHeight * _kernelWidth;
    
    _kernelMatrixRow = _inputChannelX4;
    _kernelMatrixCol = _outputChannelX4 * _kernelHeight * _kernelWidth;
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatKernel:(void*)floatKernel {
    void *realKernel = NULL;
    
#if defined(real_is_half)
    realKernel = malloc(sizeof(type) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
    
    convertFloat32ToFloat16(floatKernel, realKernel, _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
#elif defined(real_is_float)
    realKernel = floatKernel;
#endif
    
    _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    BROU(TransposeMatrix)(realKernel,
                          _outputChannel * _kernelHeight * _kernelWidth,
                          _inputChannel,
                          _kernel.contents,
                          _kernelMatrixRow,
                          _kernelMatrixCol);
    
#if defined(real_is_half)
    free(realKernel);
#endif
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatBias:(void*)floatBias {
    if (NULL == floatBias) {
        _haveBias = false;
        
        /**if the MTLBuffer is nil or length is 0, the Metal will be crash, don't why*/
        _bias = [device newBufferWithLength:sizeof(type)
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];;
        
        return;
    }
    
    void *realBias = NULL;
    
#if defined(real_is_half)
    /**the real is half */
    realBias = malloc(sizeof(type) * _outputChannel);
    
    convertFloat32ToFloat16(floatBias,
                            realBias,
                            _outputChannel);
#elif defined(real_is_float)
    /**the real is float*/
    realBias = floatBias;
#endif
    
    _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    memcpy(_bias.contents, realBias, sizeof(type) * _outputChannel);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    [super configShapeWithDevice:device];

    _input2MatrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                             options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _matrixMultipyShape = [device newBufferWithLength:sizeof(TensorShape)
                                              options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _matrix2OutputShape = [device newBufferWithLength:sizeof(TensorShape)
                                              options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
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

- (void)configMetalShapeWithInputShape:(TensorShape)inputShape outputShape:(TensorShape)outputShape {
    _inputMatrixRow = _inputChannelX4;
    _inputMatrixCol = (inputShape.dim1 * inputShape.dim0 + 3) / 4 * 4;
    
    _mediateMatrixRow = _outputChannelX4 * _kernelHeight * _kernelWidth;
    _mediateMatrixCol = _inputMatrixCol;
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = inputShape.dim0;
    inputShapeRef->dim1 = inputShape.dim1;
    inputShapeRef->dim2 = inputShape.dim2;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = outputShape.dim0;
    outputShapeRef->dim1 = outputShape.dim1;
    outputShapeRef->dim2 = outputShape.dim2;
    
    TensorShape *input2MatrixShapeRef = (TensorShape*)_input2MatrixShape.contents;
    input2MatrixShapeRef->dim0 = inputShape.dim1 * inputShape.dim0;
    input2MatrixShapeRef->dim1 = _inputMatrixRow;
    input2MatrixShapeRef->dim2 = _inputMatrixCol;
    
    TensorShape *matrixMultipyShapeRef = (TensorShape*)_matrixMultipyShape.contents;
    matrixMultipyShapeRef->dim0  = _mediateMatrixRow;
    matrixMultipyShapeRef->dim1  = _kernelMatrixRow;
    matrixMultipyShapeRef->dim2  = _mediateMatrixCol;
    
    TensorShape *matrix2OutputShapeRef = (TensorShape*)_matrix2OutputShape.contents;
    matrix2OutputShapeRef->dim0 = _mediateMatrixRow;
    matrix2OutputShapeRef->dim1 = _mediateMatrixCol;
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    [self checkParamsWithInputShape:inputShape outputShape:outputShape];
    [self configMetalShapeWithInputShape:inputShape outputShape:outputShape];
    
    /**malloc memory*/
    if (!_inputMatrix || _inputMatrix.length < sizeof(type) * _inputMatrixRow * _inputMatrixCol) {
        _inputMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _inputMatrixRow * _inputMatrixCol
                                                         options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    }
    
    if (_mediateMatrix || _mediateMatrix.length < sizeof(type) * _mediateMatrixRow * _mediateMatrixCol) {
        _mediateMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _mediateMatrixRow * _mediateMatrixCol
                                                           options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    }
    
    id<MTLComputeCommandEncoder> input2MatrixEncoder = [commandBuffer computeCommandEncoder];
    [input2MatrixEncoder setComputePipelineState:_input2MatrixComputePipelineState];
    [input2MatrixEncoder setBuffer:input              offset:0 atIndex:0];
    [input2MatrixEncoder setBuffer:_inputMatrix       offset:0 atIndex:1];
    [input2MatrixEncoder setBuffer:_input2MatrixShape offset:0 atIndex:2];
    
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((_inputMatrixCol + 31) / 32,
                                (_inputMatrixRow + 15) / 16,
                                1);
    
    [input2MatrixEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [input2MatrixEncoder endEncoding];
    
    id<MTLComputeCommandEncoder> matrixMultipyEncoder = [commandBuffer computeCommandEncoder];
    [matrixMultipyEncoder setComputePipelineState:_matrixMultiplyComputePipelineState];
    [matrixMultipyEncoder setBuffer:_kernel             offset:0 atIndex:0];
    [matrixMultipyEncoder setBuffer:_inputMatrix        offset:0 atIndex:1];
    [matrixMultipyEncoder setBuffer:_mediateMatrix      offset:0 atIndex:2];
    [matrixMultipyEncoder setBuffer:_matrixMultipyShape offset:0 atIndex:3];
    
    
    group = MTLSizeMake(8, 4, 1);
    grid   = MTLSizeMake((_mediateMatrixCol + 31) / 32,
                         (_mediateMatrixRow + 15) / 16,
                         1);
    [matrixMultipyEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [matrixMultipyEncoder endEncoding];
    
    id<MTLComputeCommandEncoder> matrix2OutputEncoder = [commandBuffer computeCommandEncoder];
    [matrix2OutputEncoder setComputePipelineState:_matrix2OutputComputePipelineState];
    [matrix2OutputEncoder setBuffer:_mediateMatrix         offset:0 atIndex:0];
    [matrix2OutputEncoder setBuffer:output                 offset:0 atIndex:1];
    [matrix2OutputEncoder setBuffer:_bias                  offset:0 atIndex:2];
    [matrix2OutputEncoder setBuffer:_matrix2OutputShape    offset:0 atIndex:3];
    [matrix2OutputEncoder setBuffer:_inputShape            offset:0 atIndex:4];
    [matrix2OutputEncoder setBuffer:_outputShape           offset:0 atIndex:5];
    [matrix2OutputEncoder setBuffer:_convolutionShape      offset:0 atIndex:6];
    
    group = MTLSizeMake(8, 4, 1);
    grid   = MTLSizeMake((outputShape.dim1 + 31) / 32,
                         (outputShape.dim0 + 15) / 16,
                         outputShape.dim2);
    
    [matrix2OutputEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [matrix2OutputEncoder endEncoding];
}
@end

#endif









