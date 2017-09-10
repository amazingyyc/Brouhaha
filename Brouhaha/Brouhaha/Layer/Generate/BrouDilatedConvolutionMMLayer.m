#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(DilatedConvolutionMMLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   floatKernel:(void *)floatKernel
                     floatBias:(void *)floatBias
                  inputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX
                       dilateY:(int)dilatedY
                      dilatedX:(int)dilatedX {
    self = [super initWithName:@BROU_OBJECT_NAME(DilatedConvolutionMMLayer)];
    
    if (!self) {
        return self;
    }
    
    [self configParamsWithInputChannel:inputChannel
                         outputChannel:outputChannel
                          kernelHeight:kernelHeight
                           kernelWidth:kernelWidth
                                padTop:padTop
                               padLeft:padLeft
                               strideY:strideY
                               strideX:strideX
                              dilatedY:dilatedY
                              dilatedX:dilatedX];
    
    [self configBufferWithDevice:device floatKernel:floatKernel];
    [self configBufferWithDevice:device floatBias:floatBias];
    [self configShapeWithDevice:device];
    
    _convertInputFunctionName = @BROU_METAL(DilatedConvolutionInput2Matrix);
    
    if (_haveBias) {
        _functionName = @BROU_METAL(MatrixMultiplyWithBias);
    } else {
        _functionName = @BROU_METAL(MatrixMultiply);
    }
    
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configParamsWithInputChannel:(int)inputChannel
                       outputChannel:(int)outputChannel
                        kernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX
                            dilatedY:(int)dilatedY
                            dilatedX:(int)dilatedX {
    [super configParamsWithInputChannel:inputChannel
                          outputChannel:outputChannel
                           kernelHeight:kernelHeight
                            kernelWidth:kernelWidth
                                 padTop:padTop
                                padLeft:padLeft
                                strideY:strideY
                                strideX:strideX
                               dilatedY:dilatedY
                               dilatedX:dilatedX];
    
    _kernelHeightXkernelWidthXinputChannelX4 = _kernelHeight * _kernelWidth * _inputChannelX4;
}

/**
 * config the buffer
 */
- (void)configBufferWithDevice:(id<MTLDevice>)device floatKernel:(void*)floatKernel {
    void *realKernel = NULL;
    
#if defined(real_is_half)
    void *tempHalfKernel = malloc(sizeof(type) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
    convertFloat32ToFloat16(floatKernel,
                            tempHalfKernel,
                            _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
    
    if (_inputChannel == _inputChannelX4) {
        realKernel = tempHalfKernel;
    } else {
        realKernel = calloc(_outputChannel * _kernelHeight * _kernelWidth * _inputChannelX4, sizeof(type));
        
        for (int c = 0; c < _outputChannel; ++c) {
            void *realKernelOffset = realKernel + c * _kernelHeight * _kernelWidth * _inputChannelX4 * sizeof(type);
            void *tempHalfKernelOffset = tempHalfKernel + c * _kernelHeight * _kernelWidth * _inputChannel * sizeof(type);
            
            for (int i = 0; i < _kernelHeight * _kernelWidth; ++i) {
                memcpy(realKernelOffset, tempHalfKernelOffset, _inputChannel * sizeof(type));
                
                realKernelOffset     += _inputChannelX4 * sizeof(type);
                tempHalfKernelOffset += _inputChannel   * sizeof(type);
            }
        }
        
        free(tempHalfKernel);
    }
#elif defined(real_is_float)
    /**the real is float*/
    
    if (_inputChannel == _inputChannelX4) {
        realKernel = floatKernel;
    } else {
        realKernel = calloc(_outputChannel * _kernelHeight * _kernelWidth * _inputChannelX4, sizeof(type));
        
        for (int c = 0; c < _outputChannel; ++c) {
            void *realKernelOffset  = realKernel  + c * _kernelHeight * _kernelWidth * _inputChannelX4 * sizeof(type);
            void *floatKernelOffset = floatKernel + c * _kernelHeight * _kernelWidth * _inputChannel   * sizeof(type);
            
            for (int i = 0; i < _kernelHeight * _kernelWidth; ++i) {
                memcpy(realKernelOffset, floatKernelOffset, _inputChannel * sizeof(type));
                
                realKernelOffset  += _inputChannelX4 * sizeof(type);
                floatKernelOffset += _inputChannel   * sizeof(type);
            }
        }
    }
#endif
    
    /**init kernel buffer*/
    _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    BROU(TransposeMatrix)(realKernel,
                          _outputChannel,
                          _kernelHeight * _kernelWidth * _inputChannelX4,
                          _kernel.contents,
                          _kernelHeight * _kernelWidth * _inputChannelX4,
                          _outputChannelX4);
    
#if defined(real_is_half)
    free(realKernel);
#elif defined(real_is_float)
    if (_inputChannel != _inputChannelX4) {
        free(realKernel);
    }
#endif
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatBias:(void*)floatBias {
    if (NULL == floatBias) {
        _haveBias = false;
        
        return;
    }
    
    _haveBias = true;
    
    void *realBias = NULL;
    
#if defined(real_is_half)
    /**the real is half */
    realBias = malloc(sizeof(type) * _outputChannel);
    
    convertFloat32ToFloat16(floatBias, realBias, _outputChannel);
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
    
    _matrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    id<MTLFunction> convertFunction = [library newFunctionWithName:_convertInputFunctionName];
    
    NSAssert(convertFunction, @"init %@ function:%@ error!", _name, _convertInputFunctionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _convertInputComputePipelineState = [device newComputePipelineStateWithFunction:convertFunction error:&error];
    
    NSAssert(_convertInputComputePipelineState, @"init %@ ConvertInputComputePipelineState error:%@", _name, error);
    
    [super configComputePipelinesStateWithDevice:device library:library];
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    /**check params*/
    [super checkParamsWithInputShape:inputShape outputShape:outputShape];
    
    int _outputHeightXoutputWidth   = outputShape.dim0 * outputShape.dim1;
    int _outputHeightXoutputWidthX4 = (_outputHeightXoutputWidth + 3) / 4 * 4;
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = inputShape.dim0;
    inputShapeRef->dim1 = inputShape.dim1;
    inputShapeRef->dim2 = inputShape.dim2;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = outputShape.dim0;
    outputShapeRef->dim1 = outputShape.dim1;
    outputShapeRef->dim2 = outputShape.dim2;
    
    TensorShape *matrixShapeRef = (TensorShape*)_matrixShape.contents;
    matrixShapeRef->dim0 = _outputHeightXoutputWidthX4;
    matrixShapeRef->dim1 = _kernelHeightXkernelWidthXinputChannelX4;
    matrixShapeRef->dim2 = _outputChannelX4;
    
    if (!_inputMatrix || _inputMatrix.length < sizeof(type) * _outputHeightXoutputWidthX4 * _kernelHeightXkernelWidthXinputChannelX4) {
        _inputMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _outputHeightXoutputWidthX4 * _kernelHeightXkernelWidthXinputChannelX4
                                                         options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    }
    
    /**config convert input to matrix encoder*/
    id<MTLComputeCommandEncoder> convertInputEncode = [commandBuffer computeCommandEncoder];
    [convertInputEncode setComputePipelineState:_convertInputComputePipelineState];
    [convertInputEncode setBuffer:input             offset:0 atIndex:0];
    [convertInputEncode setBuffer:_inputMatrix      offset:0 atIndex:1];
    [convertInputEncode setBuffer:_inputShape       offset:0 atIndex:2];
    [convertInputEncode setBuffer:_outputShape      offset:0 atIndex:3];
    [convertInputEncode setBuffer:_convolutionShape offset:0 atIndex:4];
    
    MTLSize group = MTLSizeMake(32, 1, 1);
    MTLSize grid  = MTLSizeMake((_outputHeightXoutputWidthX4  + 127) / 128,
                                1,
                                1);
    
    [convertInputEncode dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [convertInputEncode endEncoding];
    
    /**matrix multiply*/
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    if (_haveBias) {
        [encoder setBuffer:_inputMatrix         offset:0 atIndex:0];
        [encoder setBuffer:_kernel              offset:0 atIndex:1];
        [encoder setBuffer:output               offset:0 atIndex:2];
        [encoder setBuffer:_bias                offset:0 atIndex:3];
        [encoder setBuffer:_matrixShape         offset:0 atIndex:4];
    } else {
        [encoder setBuffer:_inputMatrix         offset:0 atIndex:0];
        [encoder setBuffer:_kernel              offset:0 atIndex:1];
        [encoder setBuffer:output               offset:0 atIndex:2];
        [encoder setBuffer:_matrixShape         offset:0 atIndex:3];
    }
    
    /**
     * every thread will handle 4X4X4 output
     */
    group = MTLSizeMake(8, 4, 1);
    grid  = MTLSizeMake((_outputChannelX4 + 31) / 32,
                        (_outputHeightXoutputWidthX4 + 15) / 16,
                        1);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif










