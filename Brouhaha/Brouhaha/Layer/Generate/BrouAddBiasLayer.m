#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(AddBiasLayer)() {
    /**the tensor shape*/
    DimensionType _dimensionType;
    
    NSString *_functionName;
    
    /**the Metal computePipelineState*/
    id<MTLComputePipelineState> _computePipelineState;
    
    id<MTLBuffer> _bias;
    id<MTLBuffer> _shape;
    
    int _biasLength;
    int _biasLengthX4;
}

@end

@implementation BROU_OBJECT(AddBiasLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                     floatBias:(void*)floatBias
                    biasLength:(int)biasLength
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithName:@BROU_OBJECT_NAME(AddBiasLayer)];
    
    if (!self) {
        return self;
    }
    
    NSAssert(biasLength > 0, @"the bias length must > 0");
    NSAssert(floatBias, @"the bias can not be null");
    
    _biasLength    = biasLength;
    _biasLengthX4  = (_biasLength + 3) / 4 * 4;
    _dimensionType = dimensionType;
    
    if (@available(iOS 9.0, *)) {
        _shape = [device newBufferWithLength:sizeof(TensorShape)
                                     options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _shape = [device newBufferWithLength:sizeof(TensorShape)
                                     options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    [self configBufferWithDevice:device floatBias:floatBias];
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatBias:(void*)floatBias {
    void *realBias = NULL;
    
#if defined(real_is_half)
    realBias = malloc(sizeof(type) * _biasLength);
    
    convertFloat32ToFloat16(floatBias, realBias, _biasLength);
#elif defined(real_is_float)
    realBias = floatBias;
#endif
    
    if (@available(iOS 9.0, *)) {
        _bias = [device newBufferWithLength:sizeof(type) * _biasLengthX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _bias = [device newBufferWithLength:sizeof(type) * _biasLengthX4
                                    options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    memcpy(_bias.contents, realBias, sizeof(type) * _biasLength);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    if (Dimension1D == _dimensionType) {
        _functionName = @BROU_METAL(AddBias1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(AddBias2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(AddBias3D);
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
        NSAssert(_biasLength == input.dim0, @"the input length is error");
        NSAssert(input.dim0 == output.dim0, @"the input length must equal to output length");
    } else if (Dimension2D == _dimensionType) {
        NSAssert(input.dim0 == output.dim0 && input.dim1 == output.dim1, @"the input dim must equal to ouput dim");
        NSAssert(_biasLength == input.innermostDim && _biasLength == output.innermostDim, @"the input/output dim is error");
    } else if (Dimension3D == _dimensionType) {
        NSAssert(input.dim0 == output.dim0 && input.dim1 == output.dim1 && input.dim2 == output.dim2,
                 @"the input dim must equal to ouput dim");
        NSAssert(_biasLength == input.innermostDim && _biasLength == output.innermostDim, @"the input/output dim is error");
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
    [encoder setBuffer:_bias               offset:0 atIndex:1];
    [encoder setBuffer:output.tensorBuffer offset:0 atIndex:2];
    [encoder setBuffer:_shape              offset:0 atIndex:3];
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif
