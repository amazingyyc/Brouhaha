#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(DilatedConvolutionLayer)() {
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
    
    int _dilatedY;
    int _dilatedX;
    
    /**
     * _inputChannelX4 >= inputchannel and timed by 4
     * _inputChannelX4 >= outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;
    
    /**if the convolution has a bias*/
    bool _haveBias;
    
    /**store the kernel and bias*/
    id<MTLBuffer> _kernel;
    id<MTLBuffer> _bias;
    
    /**store the params and dimension of input/output*/
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
    
    /**the MTL function name*/
    NSString *_functionName;
    
    /**the Metal computePipelineState*/
    id<MTLComputePipelineState> _computePipelineState;
}

@end

@implementation BROU_OBJECT(DilatedConvolutionLayer)

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
    self = [super initWithName:@BROU_OBJECT_NAME(DilatedConvolutionLayer)];
    
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
    NSAssert(inputChannel  > 0, @"the inputchannel must > 0");
    NSAssert(outputChannel > 0, @"the outputChannel must > 0");
    NSAssert(kernelHeight  > 0, @"the kernelHeight must > 0");
    NSAssert(kernelWidth   > 0, @"the kernelWidth must > 0");
    NSAssert(padTop  >= 0, @"the padTop must >= 0");
    NSAssert(padLeft >= 0, @"the padLeft must >= 0");
    NSAssert(strideY  > 0, @"the strideY must > 0");
    NSAssert(strideX  > 0, @"the padLeft must > 0");
    NSAssert(dilatedY > 0, @"the dilatedY must > 0");
    NSAssert(dilatedX > 0, @"the dilatedX must > 0");
    
    _inputChannel  = inputChannel;
    _outputChannel = outputChannel;
    _kernelHeight  = kernelHeight;
    _kernelWidth   = kernelWidth;
    
    _padTop  = padTop;
    _padLeft = padLeft;
    _strideY = strideY;
    _strideX = strideX;
    
    _dilatedY = dilatedY;
    _dilatedX = dilatedX;
    
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
    realKernel = malloc(sizeof(type) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
    
    convertFloat32ToFloat16(floatKernel,
                            realKernel,
                            _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
#elif defined(real_is_float)
    /**the real is float*/
    realKernel = floatKernel;
#endif
    
    /**init kernel buffer*/
    if (@available(iOS 9.0, *)) {
        _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache];
    }
    
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
        
        /**if the MTLBuffer is nil or length is 0, the Metal will be crash, don't know why???*/
        if (@available(iOS 9.0, *)) {
            _bias = [device newBufferWithLength:4
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        } else {
            _bias = [device newBufferWithLength:4
                                        options:MTLResourceCPUCacheModeDefaultCache];
        }
        
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
    
    if (@available(iOS 9.0, *)) {
        _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache];    }
    
    memcpy(_bias.contents, realBias, sizeof(type) * _outputChannel);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    if (@available(iOS 9.0, *)) {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    if (@available(iOS 9.0, *)) {
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    if (@available(iOS 9.0, *)) {
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
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
    convolutionShapeRef->dilatedY     = _dilatedY;
    convolutionShapeRef->dilatedX     = _dilatedX;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    /**init function name*/
    _functionName = @BROU_METAL(DilatedConvolution);
    
    id<MTLFunction> function = [library newFunctionWithName:_functionName];
    
    NSAssert(function, @"init %@ function:%@ error!", self.name, _functionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    
    NSAssert(_computePipelineState, @"init %@ ComputePipelineState error:%@", self.name, error);
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
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = input.dim0;
    inputShapeRef->dim1 = input.dim1;
    inputShapeRef->dim2 = input.innermostDimX4;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = output.dim0;
    outputShapeRef->dim1 = output.dim1;
    outputShapeRef->dim2 = output.innermostDimX4;
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input.tensorBuffer   offset:0 atIndex:0];
    [encoder setBuffer:_kernel              offset:0 atIndex:1];
    [encoder setBuffer:_bias                offset:0 atIndex:2];
    [encoder setBuffer:output.tensorBuffer  offset:0 atIndex:3];
    [encoder setBuffer:_inputShape          offset:0 atIndex:4];
    [encoder setBuffer:_outputShape         offset:0 atIndex:5];
    [encoder setBuffer:_convolutionShape    offset:0 atIndex:6];
    
    /**
     * every thread will handle 4X4X4 output
     */
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((output.dim1 + 31) / 32,
                                (output.dim0 + 15) / 16,
                                output.innermostDimX4 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif







