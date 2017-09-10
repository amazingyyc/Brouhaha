#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(FullConnectLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  floatWeights:(void*)floatWeight
                     floatBias:(void*)floatBias
                 intputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel {
    self = [super initWithName:@BROU_OBJECT_NAME(FullConnectLayer)];
    
    if (!self) {
        return self;
    }
    
    [self configParamsWithInputChannel:inputChannel outputChannel:outputChannel];
    [self configBufferWithDevice:device floatKernel:floatWeight];
    [self configBufferWithDevice:device floatBias:floatBias];
    
    if (_haveBias) {
        _functionName = @BROU_METAL(Fullconnect);
    } else {
        _functionName = @BROU_METAL(FullconnectWithoutBias);
    }
    
    [self configComputePipelinesStateWithDevice:device library:library];
    
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    shapeRef->dim0 = _inputChannelX4;
    shapeRef->dim1 = _outputChannelX4;
    
    return self;
}

- (void)configParamsWithInputChannel:(int)inputChannel
                       outputChannel:(int)outputChannel {
    NSAssert(inputChannel  > 0, @"the inputchannel must > 0");
    NSAssert(outputChannel > 0, @"the outputChannel must > 0");
    
    _inputChannel  = inputChannel;
    _outputChannel = outputChannel;
    
    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatKernel:(void*)floatKernel {
    void *realKernel = NULL;
    
#if defined(real_is_half)
    realKernel = malloc(sizeof(type) * _outputChannel * _inputChannel);
    
    convertFloat32ToFloat16(floatKernel,
                            realKernel,
                            _outputChannel * _inputChannel);
#elif defined(real_is_float)
    realKernel = floatKernel;
#endif
    
    _weigths = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _inputChannelX4
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    memset(_weigths.contents, 0, sizeof(type) * _outputChannelX4 * _inputChannelX4);
    
    for (int i = 0; i < _outputChannel; ++i) {
        memcpy(_weigths.contents + i * sizeof(type) * _inputChannelX4,
               realKernel        + i * sizeof(type) * _inputChannel,
               sizeof(type) * _inputChannel);
    }
    
#if defined(real_is_half)
    free(realKernel);
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
    
    convertFloat32ToFloat16(floatBias,
                            realBias,
                            _outputChannel);
#elif defined(real_is_float)
    /**the real is float*/
    realBias = floatBias;
#endif
    
    _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    memset(_bias.contents, 0, sizeof(type) * _outputChannelX4);
    memcpy(_bias.contents, realBias, sizeof(type) * _outputChannel);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    NSAssert(inputShape.dim0 == _inputChannelX4, @"the input shape is error");
    NSAssert(outputShape.dim0 == _outputChannelX4, @"the input shape is error");
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];

    if (_haveBias) {
        [encoder setBuffer:input    offset:0 atIndex:0];
        [encoder setBuffer:_weigths offset:0 atIndex:1];
        [encoder setBuffer:_bias    offset:0 atIndex:2];
        [encoder setBuffer:output   offset:0 atIndex:3];
        [encoder setBuffer:_shape   offset:0 atIndex:4];
    } else {
        [encoder setBuffer:input    offset:0 atIndex:0];
        [encoder setBuffer:_weigths offset:0 atIndex:1];
        [encoder setBuffer:output   offset:0 atIndex:2];
        [encoder setBuffer:_shape   offset:0 atIndex:3];
    }
    
    NSUInteger executeWidth = _computePipelineState.threadExecutionWidth;
    
    MTLSize group = MTLSizeMake(executeWidth, 1, 1);
    MTLSize grid  = MTLSizeMake((_outputChannelX4 + 4 * executeWidth - 1) / (4 * executeWidth), 1, 1);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif










