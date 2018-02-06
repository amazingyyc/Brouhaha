#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(ConvolutionMMLayer)() {
    /**the input features channel*/
    int _inputChannel;
    
    /**the output feature channel*/
    int _outputChannel;
    
    /**the kernel dimesnion is (outputChannel, kernelHeight, kernelWidth, inputChannel)*/
    int _kernelHeight;
    int _kernelWidth;
    
    /**the pad*/
    int _padLeft;
    int _padTop;
    
    /**stride of kernel*/
    int _strideX;
    int _strideY;
    
    /**
     * _inputChannelX4 >= inputchannel and timed by 4
     * _inputChannelX4 >= outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;
    
    /**if the convolution has a bias*/
    bool _haveBias;
    
    /**_kernelHeightXkernelWidthXinputChannelX4 = kernelHeight * kernelWidth * inputChannel4*/
    int _kernelHeightXkernelWidthXinputChannelX4;
    
    /**store the kernel and bias*/
    id<MTLBuffer> _kernel;
    id<MTLBuffer> _bias;
    
    /**store the matrix multipy shape*/
    id<MTLBuffer> _matrixShape;
    
    /**store the input matrix*/
    id<MTLBuffer> _inputMatrix;
    
    /**store the params and dimension of input/output*/
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
    
    /**the fucntion name of onvert input to a matrix*/
    NSString *_convertInputFunctionName;
    
    /**the MTL function name*/
    NSString *_calculateFunctionName;
    
    /**the pipe state*/
    id<MTLComputePipelineState> _convertInputComputePipelineState;
    
    /**the Metal computePipelineState*/
    id<MTLComputePipelineState> _calculateComputePipelineState;
}

@end

@implementation BROU_OBJECT(ConvolutionMMLayer)

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
    self = [super initWithName:@BROU_OBJECT_NAME(ConvolutionMMLayer)];
    
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
    [self configShapeWith:device];
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
    if (@available(iOS 9.0, *)) {
        _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache];
    }
    
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
    
    if (@available(iOS 9.0, *)) {
        _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    memcpy(_bias.contents, realBias, sizeof(type) * _outputChannel);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)configShapeWith:(id<MTLDevice>)device {
    if (@available(iOS 9.0, *)) {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _matrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache];
        
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache];
        
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache];
        
        _matrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache];
    }

    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    convolutionShapeRef->padTop       = _padTop;
    convolutionShapeRef->padLeft      = _padLeft;
    convolutionShapeRef->strideY      = _strideY;
    convolutionShapeRef->strideX      = _strideX;
    convolutionShapeRef->haveBias     = _haveBias;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    _convertInputFunctionName = @BROU_METAL(ConvolutionInput2Matrix);
    
    if (_haveBias) {
        _calculateFunctionName = @BROU_METAL(MatrixMultiplyWithBias);
    } else {
        _calculateFunctionName = @BROU_METAL(MatrixMultiply);
    }
    
    id<MTLFunction> convertFunction = [library newFunctionWithName:_convertInputFunctionName];
    
    NSAssert(convertFunction, @"init %@ function:%@ error!", self.name, _convertInputFunctionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _convertInputComputePipelineState = [device newComputePipelineStateWithFunction:convertFunction error:&error];
    
    NSAssert(_convertInputComputePipelineState, @"init %@ ConvertInputComputePipelineState error:%@", self.name, error);
    
    id<MTLFunction> calculateFunction = [library newFunctionWithName:_calculateFunctionName];
    
    NSAssert(calculateFunction, @"init %@ function:%@ error!", self.name, _calculateFunctionName);
    
    _calculateComputePipelineState = [device newComputePipelineStateWithFunction:calculateFunction error:&error];
    
    NSAssert(_calculateComputePipelineState, @"init %@ ComputePipelineState error:%@", self.name, error);
}

- (void)checkParamsWithInput:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    NSAssert(3 == input.dimension, @"The input tensor's dimension must be 3");
    NSAssert(input.height > 0 && input.width > 0 && input.channel > 0, @"the dim of input must > 0");
    NSAssert(_inputChannelX4  == input.innermostDimX4, @"the input shape is error!");
    
    NSAssert(3 == output.dimension, @"The output tensor's dimension must be 3");
    NSAssert(output.height > 0 && output.width > 0 && output.channel > 0, @"the dim of output must > 0");
    NSAssert(_outputChannelX4  == output.innermostDimX4, @"the output shape is error!");
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput:input output:output];
    
    int _outputHeightXoutputWidth   = output.dim0 * output.dim1;
    int _outputHeightXoutputWidthX4 = (_outputHeightXoutputWidth + 3) / 4 * 4;
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = input.dim0;
    inputShapeRef->dim1 = input.dim1;
    inputShapeRef->dim2 = input.innermostDimX4;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = output.dim0;
    outputShapeRef->dim1 = output.dim1;
    outputShapeRef->dim2 = output.innermostDimX4;
    
    TensorShape *matrixShapeRef = (TensorShape*)_matrixShape.contents;
    matrixShapeRef->dim0 = _outputHeightXoutputWidthX4;
    matrixShapeRef->dim1 = _kernelHeightXkernelWidthXinputChannelX4;
    matrixShapeRef->dim2 = _outputChannelX4;
    
    if (!_inputMatrix ||
        _inputMatrix.length < sizeof(type) * _outputHeightXoutputWidthX4 * _kernelHeightXkernelWidthXinputChannelX4) {
        if (@available(iOS 9.0, *)) {
            _inputMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _outputHeightXoutputWidthX4 * _kernelHeightXkernelWidthXinputChannelX4
                                                             options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        } else {
            _inputMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _outputHeightXoutputWidthX4 * _kernelHeightXkernelWidthXinputChannelX4
                                                             options:MTLResourceCPUCacheModeDefaultCache];
        }
    }
    
    /**config convert input to matrix encoder*/
    id<MTLComputeCommandEncoder> convertInputEncode = [commandBuffer computeCommandEncoder];
    [convertInputEncode setComputePipelineState:_convertInputComputePipelineState];
    [convertInputEncode setBuffer:input.tensorBuffer offset:0 atIndex:0];
    [convertInputEncode setBuffer:_inputMatrix       offset:0 atIndex:1];
    [convertInputEncode setBuffer:_inputShape        offset:0 atIndex:2];
    [convertInputEncode setBuffer:_outputShape       offset:0 atIndex:3];
    [convertInputEncode setBuffer:_convolutionShape  offset:0 atIndex:4];
    
    MTLSize group = MTLSizeMake(32, 1, 1);
    MTLSize grid  = MTLSizeMake((_outputHeightXoutputWidthX4  + 127) / 128,
                                1,
                                1);
    
    [convertInputEncode dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [convertInputEncode endEncoding];
    
    /**matrix multiply*/
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_calculateComputePipelineState];
    
    if (_haveBias) {
        [encoder setBuffer:_inputMatrix         offset:0 atIndex:0];
        [encoder setBuffer:_kernel              offset:0 atIndex:1];
        [encoder setBuffer:output.tensorBuffer  offset:0 atIndex:2];
        [encoder setBuffer:_bias                offset:0 atIndex:3];
        [encoder setBuffer:_matrixShape         offset:0 atIndex:4];
    } else {
        [encoder setBuffer:_inputMatrix         offset:0 atIndex:0];
        [encoder setBuffer:_kernel              offset:0 atIndex:1];
        [encoder setBuffer:output.tensorBuffer  offset:0 atIndex:2];
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






