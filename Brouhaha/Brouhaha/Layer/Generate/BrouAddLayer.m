#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(AddLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithName:@BROU_OBJECT_NAME(AddLayer)];
    
    if (!self) {
        return self;
    }
    
    _dimensionType = dimensionType;
    
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    if (Dimension1D == _dimensionType) {
        _functionName = @BROU_METAL(Add1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(Add2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(Add3D);
    } else {
        NSAssert(false, @"the dimension type is error");
    }
    
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)checkParamsWithShape:(TensorShape)shape {
    if (Dimension1D == _dimensionType) {
        NSAssert(   shape.dim0 > 0
                 && 0 == shape.dim0 % 4,
                 @"the input length must > 0 and timed by 4");
    } else if (Dimension2D == _dimensionType) {
        NSAssert(   shape.dim0 > 0
                 && shape.dim1 > 0
                 && 0 == shape.dim1 % 4,
                 @"the input heigth, width must > 0 and width timed by4");
    } else if (Dimension3D == _dimensionType) {
        NSAssert(   shape.dim0 > 0
                 && shape.dim1 > 0
                 && shape.dim2 > 0
                 && 0 == shape.dim2 % 4,
                 @"the input height, width, channel must > 0 and channel timed by4");
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    NSAssert(false, @"the add layer should use the function:computeWithCommandBuffer:input1:input2:output:shape");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                          input1:(id<MTLBuffer>)input1
                          input2:(id<MTLBuffer>)input2
                          output:(id<MTLBuffer>)output
                           shape:(TensorShape)shape {
    [self checkParamsWithShape:shape];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    shapeRef->dim0 = shape.dim0;
    shapeRef->dim1 = shape.dim1;
    shapeRef->dim2 = shape.dim2;
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    [encoder setBuffer:input1 offset:0 atIndex:0];
    [encoder setBuffer:input2 offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBuffer:_shape offset:0 atIndex:3];
    
    /**
     * every thread will handle 4X4X4 output
     */
    NSUInteger exeWidth = _computePipelineState.threadExecutionWidth;
    MTLSize group = MTLSizeMake(1, 1, 1);
    MTLSize grid  = MTLSizeMake(1, 1, 1);
    
    if (Dimension1D == _dimensionType) {
        group = MTLSizeMake(exeWidth, 1, 1);
        grid  = MTLSizeMake((shape.dim0 + exeWidth * 4 - 1) / (exeWidth * 4),
                            1,
                            1);
    } else if (Dimension2D == _dimensionType) {
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((shape.dim1 + 31) / 32,
                            (shape.dim0 + 15) / 16,
                            1);
    } else if (Dimension3D == _dimensionType) {
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((shape.dim1 + 31) / 32,
                            (shape.dim0 + 15) / 16,
                            (shape.dim2 / 4));
    } else {
        /**todo support all dimension data*/
        NSAssert(false, @"The data dimension is error");
    }
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif
