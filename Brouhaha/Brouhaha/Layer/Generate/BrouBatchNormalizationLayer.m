#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(BatchNormalizationLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                       epsilon:(float)epsilon
                    floatAlpha:(void*)floatAlpha
                     floatBeta:(void*)floatBeta
                       channel:(int)channel {
    self = [super initWithName:@BROU_OBJECT_NAME(BatchNormalizationLayer)];
    
    if (!self) {
        return self;
    }
    
    NSAssert(channel > 0, @"channel must be > 0");
    
    _channel = channel;
    _channelX4 = (_channel + 3) / 4 * 4;
    
    _floatEpison = epsilon;
    
    _bnShape = [device newBufferWithLength:sizeof(BatchNormalizationShape)
                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    BatchNormalizationShape *bnShapeRef = (BatchNormalizationShape*)_bnShape.contents;
    bnShapeRef->epsilon = _floatEpison;
    
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _mean = [device newBufferWithLength:sizeof(type) * _channelX4
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _variance = [device newBufferWithLength:sizeof(type) * _channelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    [self configBufferWithDevice:device floatAlpha:floatAlpha];
    [self configBufferWithDevice:device floatBeta:floatBeta];
    
    _calculateMeanVarianceFunctionName = @BROU_METAL(CalculateMeanAndVariance3D);
    _functionName = @BROU_METAL(BatchNormalization3D);
    
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatAlpha:(void*)floatAlpha {
    void *realAlpha;
    
#if defined(real_is_half)
    realAlpha = malloc(sizeof(type) * _channel);
    convertFloat32ToFloat16(floatAlpha, realAlpha, _channel);
#elif defined(real_is_float)
    realAlpha = floatAlpha;
#endif
    
    _alpha = [device newBufferWithLength:sizeof(type) * _channelX4
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    memcpy(_alpha.contents, realAlpha, sizeof(type) * _channel);
    
#if defined(real_is_half)
    free(realAlpha);
#endif
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatBeta:(void*)floatBeta {
    void *realBeta;
    
#if defined(real_is_half)
    realBeta = malloc(sizeof(type) * _channel);
    convertFloat32ToFloat16(floatBeta, realBeta, _channel);
#elif defined(real_is_float)
    realBeta = floatBeta;
#endif
    
    _beta = [device newBufferWithLength:sizeof(type) * _channelX4
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    memcpy(_beta.contents, realBeta, sizeof(type) * _channel);
    
#if defined(real_is_half)
    free(realBeta);
#endif
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    [super configComputePipelinesStateWithDevice:device library:library];
    
    id<MTLFunction> function = [library newFunctionWithName:_calculateMeanVarianceFunctionName];
    
    NSAssert(function, @"init %@ function:%@ error!", _name, _calculateMeanVarianceFunctionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _calculateMeanVariancePipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    
    NSAssert(_calculateMeanVariancePipelineState, @"init %@ MeanVariancePipelineState error:%@", _name, error);
}

- (void)checkParamsWithInputShape:(TensorShape)inputShape
                      outputShape:(TensorShape)outputShape {
    NSAssert(inputShape.dim0 > 0, @"the input height must > 0");
    NSAssert(inputShape.dim1 > 0, @"the input width must > 0");
    NSAssert(_channelX4  == inputShape.dim2, @"the input shape is error!");
    NSAssert(outputShape.dim0 > 0, @"the output height must > 0");
    NSAssert(outputShape.dim1 > 0, @"the output width must > 0");
    NSAssert(_channelX4 == outputShape.dim2, @"the output shape is error!");
    NSAssert(inputShape.dim0 == outputShape.dim0
             && inputShape.dim1 == outputShape.dim1, @"the shape is error!");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    [self checkParamsWithInputShape:inputShape outputShape:outputShape];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    shapeRef->dim0 = inputShape.dim0;
    shapeRef->dim1 = inputShape.dim1;
    shapeRef->dim2 = _channelX4;
    
    int perThreadWidth  = (inputShape.dim1 + 7) / 8;
    int perThreadHeight = (inputShape.dim0 + 3) / 4;
    
    BatchNormalizationShape *bnShapeRef = (BatchNormalizationShape*)_bnShape.contents;
    bnShapeRef->perThreadWidth  = perThreadWidth;
    bnShapeRef->perThreadHeight = perThreadHeight;
    
    /**calcualte mean*/
    id<MTLComputeCommandEncoder> meanVarianceEncoder = [commandBuffer computeCommandEncoder];
    [meanVarianceEncoder setComputePipelineState:_calculateMeanVariancePipelineState];
    [meanVarianceEncoder setBuffer:input      offset:0 atIndex:0];
    [meanVarianceEncoder setBuffer:_mean      offset:0 atIndex:1];
    [meanVarianceEncoder setBuffer:_variance  offset:0 atIndex:2];
    [meanVarianceEncoder setBuffer:_bnShape   offset:0 atIndex:3];
    [meanVarianceEncoder setBuffer:_shape     offset:0 atIndex:4];
    
    NSUInteger exeWidth = _calculateMeanVariancePipelineState.threadExecutionWidth;
    
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake(1,
                                1,
                                _channelX4 / 4);
    
    [meanVarianceEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [meanVarianceEncoder endEncoding];
    
    /**calcualte output*/
    id<MTLComputeCommandEncoder> bnEncoder = [commandBuffer computeCommandEncoder];
    [bnEncoder setComputePipelineState:_computePipelineState];
    [bnEncoder setBuffer:input     offset:0 atIndex:0];
    [bnEncoder setBuffer:output    offset:0 atIndex:1];
    [bnEncoder setBuffer:_mean     offset:0 atIndex:2];
    [bnEncoder setBuffer:_variance offset:0 atIndex:3];
    [bnEncoder setBuffer:_alpha    offset:0 atIndex:4];
    [bnEncoder setBuffer:_beta     offset:0 atIndex:5];
    [bnEncoder setBuffer:_shape    offset:0 atIndex:6];
    
    group = MTLSizeMake(8, 4, 1);
    grid  = MTLSizeMake((inputShape.dim1 + 31) / 32,
                        (inputShape.dim0 + 15) / 16,
                        _channelX4 / 4);
    
    [bnEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [bnEncoder endEncoding];
}

@end

#endif









