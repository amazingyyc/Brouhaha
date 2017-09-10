#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(OperateLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithName:@BROU_OBJECT_NAME(OperateLayer)];
    
    if (!self) {
        return self;
    }
    
    _dimensionType = dimensionType;
    
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    [self configFunctionNameWithDimensionType:_dimensionType];
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configFunctionNameWithDimensionType:(DimensionType)dimensionType {
    NSAssert(false, @"can not init from OperateLayer");
}

- (void)checkParamsWithInputShape:(TensorShape)inputShape
                      outputShape:(TensorShape)outputShape {
    if (Dimension1D == _dimensionType) {
        NSAssert(inputShape.dim0 == outputShape.dim0
                 && inputShape.dim0 > 0
                 && 0 == inputShape.dim0 % 4,
                 @"the input length must == output length and > 0 and timed by 4");
    } else if (Dimension2D == _dimensionType) {
        NSAssert(inputShape.dim0 == outputShape.dim0
                 && inputShape.dim0 > 0,
                 @"the input height must == output height and > 0");
        NSAssert(inputShape.dim1 == outputShape.dim1
                 && inputShape.dim1 > 0
                 && 0 == inputShape.dim1 % 4,
                 @"the input width must == output width and > 0 and timed by4");
    } else if (Dimension3D == _dimensionType) {
        NSAssert(inputShape.dim0 == outputShape.dim0
                 && inputShape.dim0 > 0,
                 @"the input height must == output height and > 0");
        NSAssert(inputShape.dim1 == outputShape.dim1
                 && inputShape.dim1 > 0,
                 @"the input width must == output height and > 0");
        NSAssert(inputShape.dim2 ==  outputShape.dim2
                 && inputShape.dim2 > 0
                 && 0 == inputShape.dim2 % 4,
                 @"the channel must be timed by 4 and inputChannel must equal to outputChannel");
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)configEncoderBufferWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                           inputBuffer:(id<MTLBuffer>)inputBuffer
                          outputBuffer:(id<MTLBuffer>)outputBuffer {
    NSAssert(false, @"can not init from OperateLayer");
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
    shapeRef->dim2 = inputShape.dim2;

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [self configEncoderBufferWithEncoder:encoder inputBuffer:input outputBuffer:output];
    
    /**
     * every thread will handle 4X4X4 output
     */
    NSUInteger exeWidth = _computePipelineState.threadExecutionWidth;
    MTLSize group = MTLSizeMake(1, 1, 1);
    MTLSize grid  = MTLSizeMake(1, 1, 1);
    
    if (Dimension1D == _dimensionType) {
        group = MTLSizeMake(exeWidth, 1, 1);
        grid  = MTLSizeMake((inputShape.dim0 + exeWidth * 4 - 1) / (exeWidth * 4),
                             1,
                             1);
    } else if (Dimension2D == _dimensionType) {
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((inputShape.dim1 + 31) / 32,
                            (inputShape.dim0 + 15) / 16,
                            1);
    } else if (Dimension3D == _dimensionType) {
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((inputShape.dim1 + 31) / 32,
                            (inputShape.dim0 + 15) / 16,
                            (inputShape.dim2 / 4));
    } else {
        /**todo support all dimension data*/
        NSAssert(false, @"The data dimension is error");
    }
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif
