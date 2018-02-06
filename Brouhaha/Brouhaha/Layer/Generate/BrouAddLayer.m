#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(AddLayer)() {
    DimensionType _dimensionType;
    
    id<MTLBuffer> _shape;
    
    NSString *_functionName;
    
    id<MTLComputePipelineState> _computePipelineState;
}

@end

@implementation BROU_OBJECT(AddLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithName:@BROU_OBJECT_NAME(AddLayer)];
    
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
        _functionName = @BROU_METAL(Add1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(Add2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(Add3D);
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

- (void)checkParamsWithInput1:(id<BrouTensor>)input1
                       input2:(id<BrouTensor>)input2
                       output:(id<BrouTensor>)output {
    if (Dimension1D == _dimensionType) {
        NSAssert(   input1.dim0 == input2.dim0
                 && input2.dim0 == output.dim0,
                 @"the input1, input2 and ouput dimension must be equal");
        NSAssert(   input1.dim0 > 0,
                 @"the input1, input2 and ouput dimension must be > 0");
    } else if (Dimension2D == _dimensionType) {
        NSAssert(   input1.dim0 == input2.dim0
                 && input2.dim0 == output.dim0
                 && input1.dim1 == input2.dim1
                 && input2.dim1 == output.dim1
                 , @"the input1, input2 and ouput dimension must be equal");
        NSAssert(   input1.dim0 > 0
                 && input1.dim1 > 0,
                 @"the input1, input2 and ouput dimension must be > 0");
    } else if (Dimension3D == _dimensionType) {
        NSAssert(   input1.dim0 == input2.dim0
                 && input2.dim0 == output.dim0
                 && input1.dim1 == input2.dim1
                 && input2.dim1 == output.dim1
                 && input1.dim2 == input2.dim2
                 && input2.dim2 == output.dim2,
                 @"the input1, input2 and ouput dimension must be equal");
        
        NSAssert(   input1.dim0 > 0
                 && input1.dim1 > 0
                 && input1.dim2 > 0,
                 @"the input1, input2 and ouput dimension must be > 0");
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    NSAssert(false, @"the add layer should use the function computeWithCommandBuffer:input1:input2:output");
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                      input1:(id<BrouTensor>)input1
                      input2:(id<BrouTensor>)input2
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput1:input1 input2:input2 output:output];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;

    MTLSize group = MTLSizeMake(1, 1, 1);
    MTLSize grid  = MTLSizeMake(1, 1, 1);
    
    if (Dimension1D == _dimensionType) {
        shapeRef->dim0 = input1.innermostDimX4;
        
        group = MTLSizeMake(32, 1, 1);
        grid  = MTLSizeMake((shapeRef->dim0 + 32 * 4 - 1) / (32 * 4),
                            1,
                            1);
    } else if (Dimension2D == _dimensionType) {
        shapeRef->dim0 = input1.dim0;
        shapeRef->dim1 = input1.innermostDimX4;
        
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                            (shapeRef->dim0 + 15) / 16,
                            1);
    } else if (Dimension3D == _dimensionType) {
        shapeRef->dim0 = input1.dim0;
        shapeRef->dim1 = input1.dim1;
        shapeRef->dim2 = input1.innermostDimX4;
        
        group = MTLSizeMake(8, 4, 1);
        grid  = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                            (shapeRef->dim0 + 15) / 16,
                            (shapeRef->dim2 / 4));
    } else {
        NSAssert(false, @"The input/output dimension is error");
    }
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input1.tensorBuffer offset:0 atIndex:0];
    [encoder setBuffer:input2.tensorBuffer offset:0 atIndex:1];
    [encoder setBuffer:output.tensorBuffer offset:0 atIndex:2];
    [encoder setBuffer:_shape              offset:0 atIndex:3];
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif
