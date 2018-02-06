#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(ReLuLayer)() {
    DimensionType _dimensionType;
    
    id<MTLBuffer> _shape;
    
    NSString *_functionName;
    
    /**the Metal computePipelineState*/
    id<MTLComputePipelineState> _computePipelineState;
}

@end

@implementation BROU_OBJECT(ReLuLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithName:@BROU_OBJECT_NAME(ReLuLayer)];
    
    if (!self) {
        return self;
    }
    
    _dimensionType = dimensionType;
    
    if (@available(iOS 9.0, *)) {
        _shape = [device newBufferWithLength:sizeof(TensorShape)
                                     options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _shape = [device newBufferWithLength:sizeof(TensorShape)
                                     options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    if (Dimension1D == _dimensionType) {
        _functionName = @BROU_METAL(ReLu1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(ReLu2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(ReLu3D);
    } else {
        NSAssert(false, @"the dimension type is error");
    }
    
    id<MTLFunction> function = [library newFunctionWithName:_functionName];
    
    NSAssert(function, @"init %@ function:%@ error!", self.name, _functionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    
    NSAssert(_computePipelineState, @"init %@ ComputePipelineState error:%@", self.name, error);
}

- (void)checkParamsWithInput:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    if (Dimension1D == _dimensionType) {
        NSAssert(input.dim0 == output.dim0,
                 @"the input dim must be equal to output");
        NSAssert(input.dim0 > 0,
                 @"the input and output dimension must be > 0");
    } else if (Dimension2D == _dimensionType) {
        NSAssert(   input.dim0 == output.dim0
                 && input.dim1 == output.dim1,
                 @"the input dim must be equal to output");
        NSAssert(   input.dim0 > 0
                 && input.dim1 > 0,
                 @"the input and output dimension must be > 0");
    } else if (Dimension3D == _dimensionType) {
        NSAssert(   input.dim0 == output.dim0
                 && input.dim1 == output.dim1
                 && input.dim2 == output.dim2,
                 @"the input dim must be equal to output");
        NSAssert(   input.dim0 > 0
                 && input.dim1 > 0
                 && input.dim2 > 0,
                 @"the input and output dimension must be > 0");
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput:input output:output];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    MTLSize group = MTLSizeMake(1, 1, 1);
    MTLSize grid  = MTLSizeMake(1, 1, 1);
    
    if (Dimension1D == _dimensionType) {
        shapeRef->dim0 = input.innermostDimX4;
        
        group = MTLSizeMake(32, 1, 1);
        grid  = MTLSizeMake((shapeRef->dim0 + 32 * 4 - 1) / (32 * 4),
                            1,
                            1);
    } else if (Dimension2D == _dimensionType) {
        shapeRef->dim0 = input.dim0;
        shapeRef->dim1 = input.innermostDimX4;
        
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                            (shapeRef->dim0 + 15) / 16,
                            1);
    } else if (Dimension3D == _dimensionType) {
        shapeRef->dim0 = input.dim0;
        shapeRef->dim1 = input.dim1;
        shapeRef->dim2 = input.innermostDimX4;
        
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                            (shapeRef->dim0 + 15) / 16,
                            (shapeRef->dim2 / 4));
    } else {
        NSAssert(false, @"The input/output dimension is error");
    }
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    [encoder setBuffer:input.tensorBuffer  offset:0 atIndex:0];
    [encoder setBuffer:output.tensorBuffer offset:0 atIndex:1];
    [encoder setBuffer:_shape              offset:0 atIndex:2];
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif
