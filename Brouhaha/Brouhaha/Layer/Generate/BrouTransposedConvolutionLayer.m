#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(TransposedConvolutionLayer)() {
    /**
     * the transposed convolution has a coperated convolution
     * the prefix _origin means the coperated convolution (origin convolution)
     * the property that without _origin mean the transpoed convolution's property
     */
    
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
    
    /**
     * the origin convolution input dimension
     */
    int _originInputChannel;
    
    /**
     * the origin convolution output dimension
     */
    int _originOutputChannnel;
    
    /**
     * the origin convoluton's kernel
     */
    int _originKernelHeight;
    int _originKernelWidth;
    
    /**
     * the origin pad
     */
    int _originPadLeft;
    int _originPadTop;
    
    /**
     * the origin stride
     */
    int _originStrideX;
    int _originStrideY;
    
    /**
     * insertX = strideX - 1
     * insertY = strideY - 1
     * insert 0-uints to the input on x/y axis
     */
    int _insertX;
    int _insertY;
    
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

@implementation BROU_OBJECT(TransposedConvolutionLayer)

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
    self = [super initWithName:@BROU_OBJECT_NAME(TransposedConvolutionLayer)];
    
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
    NSAssert(originInputChannel  > 0, @"the originInputChannel must > 0");
    NSAssert(originOutputChannel > 0, @"the originOutputChannel must > 0");
    NSAssert(originKernelHeight  > 0, @"the originKernelHeight must > 0");
    NSAssert(originKernelWidth   > 0, @"the originKernelWidth must > 0");
    NSAssert(originPadTop  >= 0, @"the originPadTop must >= 0");
    NSAssert(originPadLeft >= 0, @"the originPadLeft must >= 0");
    NSAssert(originStrideY  > 0, @"the originStrideY must > 0");
    NSAssert(originStrideX  > 0, @"the originStrideX must > 0");
    NSAssert(originKernelHeight > originPadTop, @"the originKernelHeight must > originPadTop");
    NSAssert(originKernelWidth  > originPadLeft, @"the originKernelWidth must > originPadLeft");
    
    _originInputChannel     = originInputChannel;
    _originOutputChannnel   = originOutputChannel;
    _originKernelHeight     = originKernelHeight;
    _originKernelWidth      = originKernelWidth;
    _originPadTop           = originPadTop;
    _originPadLeft          = originPadLeft;
    _originStrideY          = originStrideY;
    _originStrideX          = originStrideX;
    
    _insertX = _originStrideX - 1;
    _insertY = _originStrideY - 1;
    
    _inputChannel  = _originOutputChannnel;
    _outputChannel = _originInputChannel;
    _kernelHeight  = _originKernelHeight;
    _kernelWidth   = _originKernelWidth;
    
    _padTop  = _originKernelHeight - _originPadTop  - 1;
    _padLeft = _originKernelWidth  - _originPadLeft - 1;
    _strideY = 1;
    _strideX = 1;
    
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
    convolutionShapeRef->insertY      = _insertY;
    convolutionShapeRef->insertX      = _insertX;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    /**init function name*/
    _functionName = @BROU_METAL(TransposedConvolution);
    
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
    MTLSize grid  = MTLSizeMake((outputShapeRef->dim1 + 31) / 32,
                                (outputShapeRef->dim0 + 15) / 16,
                                 outputShapeRef->dim2 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif






