#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(ConvolutionLayer)

/**
 * this init function can init float32 and float16 layer
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   floatKernel:(void*)floatKernel
                     floatBias:(void*)floatBias
                  inputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX {
    self = [super initWithName:@BROU_OBJECT_NAME(ConvolutionLayer)];
    
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
                               strideX:strideX];

    [self configBufferWithDevice:device floatKernel:floatKernel];
    [self configBufferWithDevice:device floatBias:floatBias];
    [self configShapeWithDevice:device];
    
    /**init function name*/
    _functionName = @BROU_METAL(Convolution);
    
    /**config Metal function*/
    [self configComputePipelinesStateWithDevice:device
                                        library:library];
    
    return self;
}

- (void)configParamsWithInputChannel:(int)inputChannel
                       outputChannel:(int)outputChannel
                        kernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX {
    NSAssert(inputChannel  > 0, @"the inputchannel must > 0");
    NSAssert(outputChannel > 0, @"the outputChannel must > 0");
    NSAssert(kernelHeight  > 0, @"the kernelHeight must > 0");
    NSAssert(kernelWidth   > 0, @"the kernelWidth must > 0");
    NSAssert(padTop  >= 0, @"the padTop must >= 0");
    NSAssert(padLeft >= 0, @"the padLeft must >= 0");
    NSAssert(strideY  > 0, @"the strideY must > 0");
    NSAssert(strideX  > 0, @"the padLeft must > 0");
    
    _inputChannel  = inputChannel;
    _outputChannel = outputChannel;
    _kernelHeight  = kernelHeight;
    _kernelWidth   = kernelWidth;
    
    _padTop  = padTop;
    _padLeft = padLeft;
    _strideY = strideY;
    _strideX = strideX;
    
    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;
}

/**
 * config the buffer
 */
- (void)configBufferWithDevice:(id<MTLDevice>)device floatKernel:(void*)floatKernel {
    void *realKernel = NULL;
    
#if defined(real_is_half)
    /**the real is half */
    realKernel = malloc(sizeof(type)*_outputChannel*_kernelHeight*_kernelWidth*_inputChannel);
    
    convertFloat32ToFloat16(floatKernel,
                            realKernel,
                            _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
#elif defined(real_is_float)
    /**the real is float*/
    realKernel = floatKernel;
#endif
    
    /**init kernel buffer*/
    _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    for (int i = 0; i < _outputChannel; ++i) {
        void *bufferOffset  = _kernel.contents + sizeof(type) * i * _kernelHeight * _kernelWidth * _inputChannelX4;
        void *dataOffset    = realKernel + sizeof(type) * i * _kernelHeight * _kernelWidth * _inputChannel;
        
        for (int j = 0; j < _kernelHeight * _kernelWidth; ++j) {
            memcpy(bufferOffset, dataOffset, _inputChannel * sizeof(type));
            
            bufferOffset += _inputChannelX4 * sizeof(type);
            dataOffset   += _inputChannel   * sizeof(type);
        }
    }
    
#if defined(real_is_half)
    free(realKernel);
#endif
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatBias:(void*)floatBias {
    if (NULL == floatBias) {
        _haveBias = false;
        
        /**if the MTLBuffer is nil or length is 0, the Metal will be crash, don't why*/
        _bias = [device newBufferWithLength:4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        return;
    }
    
    _haveBias = true;
    
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
    
    // memset(_bias.contents, 0, sizeof(type) * _outputChannelX4);
    memcpy(_bias.contents, realBias, sizeof(type) * _outputChannel);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                            options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    convolutionShapeRef->padTop       = _padTop;
    convolutionShapeRef->padLeft      = _padLeft;
    convolutionShapeRef->strideY      = _strideY;
    convolutionShapeRef->strideX      = _strideX;
    convolutionShapeRef->haveBias     = _haveBias;
}

- (void)checkParamsWithInputShape:(TensorShape)inputShape
                      outputShape:(TensorShape)outputShape {
    NSAssert(inputShape.dim0 > 0, @"the input height must > 0");
    NSAssert(inputShape.dim1 > 0, @"the input width must > 0");
    NSAssert(_inputChannelX4  == inputShape.dim2, @"the input shape is error!");
    NSAssert(outputShape.dim0 > 0, @"the output height must > 0");
    NSAssert(outputShape.dim1 > 0, @"the output width must > 0");
    NSAssert(_outputChannelX4 == outputShape.dim2, @"the output shape is error!");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    /**check params*/
    [self checkParamsWithInputShape:inputShape outputShape:outputShape];
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = inputShape.dim0;
    inputShapeRef->dim1 = inputShape.dim1;
    inputShapeRef->dim2 = inputShape.dim2;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = outputShape.dim0;
    outputShapeRef->dim1 = outputShape.dim1;
    outputShapeRef->dim2 = outputShape.dim2;
    
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
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((outputShape.dim1 + 31) / 32,
                                (outputShape.dim0 + 15) / 16,
                                _outputChannelX4 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif









