#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(BatchNormalizationLayer)() {
    int _channel;
    int _channelX4;
    
    float _floatEpison;
    
    id<MTLBuffer> _bnShape;
    id<MTLBuffer> _shape;
    
    id<MTLBuffer> _mean;
    id<MTLBuffer> _variance;
    
    id<MTLBuffer> _alpha;
    id<MTLBuffer> _beta;
    
    NSString *_meanVarianceFunctionName;
    NSString *_calculateFunctionName;
    
    id<MTLComputePipelineState> _meanVariancePipelineState;
    id<MTLComputePipelineState> _calculatePipelineState;
}

@end

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

    _channel     = channel;
    _channelX4   = (_channel + 3) / 4 * 4;
    _floatEpison = epsilon;
    
    if (@available(iOS 9.0, *)) {
        _bnShape = [device newBufferWithLength:sizeof(BatchNormalizationShape)
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        _shape = [device newBufferWithLength:sizeof(TensorShape)
                                     options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        _mean = [device newBufferWithLength:sizeof(type) * _channelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        _variance = [device newBufferWithLength:sizeof(type) * _channelX4
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _bnShape  = [device newBufferWithLength:sizeof(BatchNormalizationShape) options:MTLResourceCPUCacheModeDefaultCache];
        _shape    = [device newBufferWithLength:sizeof(TensorShape) options:MTLResourceCPUCacheModeDefaultCache];
        _mean     = [device newBufferWithLength:sizeof(type) * _channelX4 options:MTLResourceCPUCacheModeDefaultCache];
        _variance = [device newBufferWithLength:sizeof(type) * _channelX4 options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    BatchNormalizationShape *bnShapeRef = (BatchNormalizationShape*)_bnShape.contents;
    bnShapeRef->epsilon = _floatEpison;
    
    [self configBufferWithDevice:device floatAlpha:floatAlpha];
    [self configBufferWithDevice:device floatBeta:floatBeta];
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
    
    if (@available(iOS 9.0, *)) {
        _alpha = [device newBufferWithLength:sizeof(type) * _channelX4
                                     options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _alpha = [device newBufferWithLength:sizeof(type) * _channelX4 options:MTLResourceCPUCacheModeDefaultCache];
    }
    
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
    
    if (@available(iOS 9.0, *)) {
        _beta = [device newBufferWithLength:sizeof(type) * _channelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _beta = [device newBufferWithLength:sizeof(type) * _channelX4 options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    memcpy(_beta.contents, realBeta, sizeof(type) * _channel);
    
#if defined(real_is_half)
    free(realBeta);
#endif
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    _meanVarianceFunctionName = @BROU_METAL(CalculateMeanAndVariance3D);
    _calculateFunctionName    = @BROU_METAL(BatchNormalization3D);
    
    id<MTLFunction> meanVarianceFunction = [library newFunctionWithName:_meanVarianceFunctionName];
    
    NSAssert(meanVarianceFunction, @"init %@ function:%@ error!", self.name, _meanVarianceFunctionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _meanVariancePipelineState = [device newComputePipelineStateWithFunction:meanVarianceFunction error:&error];
    
    NSAssert(_meanVariancePipelineState, @"init %@ MeanVariancePipelineState error:%@", self.name, error);
    
    id<MTLFunction> calculateFunction = [library newFunctionWithName:_calculateFunctionName];
    
    NSAssert(calculateFunction, @"init %@ function:%@ error!", self.name, _calculateFunctionName);
    
    _calculatePipelineState = [device newComputePipelineStateWithFunction:calculateFunction error:&error];
    
    NSAssert(_calculatePipelineState, @"init %@ ComputePipelineState error:%@", self.name, error);
}

- (void)checkParamsWithInput:(id<BrouTensor>)input output:(id<BrouTensor>)output {
    NSAssert(3 == input.dimension && input.dim0 > 0 && input.dim1 > 0 && input.dim2 > 0, @"the input dim is error");
    NSAssert(3 == output.dimension && output.dim0 > 0 && output.dim1 > 0 && output.dim2 > 0, @"the output dim is error");
    NSAssert(input.dim0 == output.dim0 && input.dim1 == output.dim1 && input.dim2 == output.dim2, @"the input and output dim must same");
    NSAssert(_channel == input.dim2, @"the input and output dim is error");
}


- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput:input output:output];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    shapeRef->dim0 = input.dim0;
    shapeRef->dim1 = input.dim1;
    shapeRef->dim2 = _channelX4;
    
    int perThreadWidth  = (shapeRef->dim1 + 7) / 8;
    int perThreadHeight = (shapeRef->dim0 + 3) / 4;
    
    BatchNormalizationShape *bnShapeRef = (BatchNormalizationShape*)_bnShape.contents;
    bnShapeRef->perThreadWidth  = perThreadWidth;
    bnShapeRef->perThreadHeight = perThreadHeight;
    
    /**calcualte mean*/
    id<MTLComputeCommandEncoder> meanVarianceEncoder = [commandBuffer computeCommandEncoder];
    [meanVarianceEncoder setComputePipelineState:_meanVariancePipelineState];
    [meanVarianceEncoder setBuffer:input.tensorBuffer   offset:0 atIndex:0];
    [meanVarianceEncoder setBuffer:_mean                offset:0 atIndex:1];
    [meanVarianceEncoder setBuffer:_variance            offset:0 atIndex:2];
    [meanVarianceEncoder setBuffer:_bnShape             offset:0 atIndex:3];
    [meanVarianceEncoder setBuffer:_shape               offset:0 atIndex:4];
    
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake(1,
                                1,
                                _channelX4 / 4);
    
    [meanVarianceEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [meanVarianceEncoder endEncoding];
    
    /**calcualte output*/
    id<MTLComputeCommandEncoder> bnEncoder = [commandBuffer computeCommandEncoder];
    [bnEncoder setComputePipelineState:_calculatePipelineState];
    [bnEncoder setBuffer:input.tensorBuffer     offset:0 atIndex:0];
    [bnEncoder setBuffer:output.tensorBuffer    offset:0 atIndex:1];
    [bnEncoder setBuffer:_mean                  offset:0 atIndex:2];
    [bnEncoder setBuffer:_variance              offset:0 atIndex:3];
    [bnEncoder setBuffer:_alpha                 offset:0 atIndex:4];
    [bnEncoder setBuffer:_beta                  offset:0 atIndex:5];
    [bnEncoder setBuffer:_shape                 offset:0 atIndex:6];
    
    group = MTLSizeMake(8, 4, 1);
    grid  = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                        (shapeRef->dim0 + 15) / 16,
                        _channelX4 / 4);
    
    [bnEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [bnEncoder endEncoding];
}


@end

#endif









