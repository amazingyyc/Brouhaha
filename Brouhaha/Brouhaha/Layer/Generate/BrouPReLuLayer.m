#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(PReLuLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                             a:(float)a
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithDevice:device library:library dimensionType:dimensionType];
    
    if (!self) {
        return self;
    }
    
    _name = @BROU_OBJECT_NAME(PReLuLayer);
    
    _aBuffer = [device newBufferWithLength:sizeof(type)
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _floatA = a;
    
#if defined(real_is_half)
    _typeA = convertFloat32ToFloat16OneNumber((uint32_t*)(&a));
#elif defined(real_is_float)
    _typeA = a;
#endif
    
    *((type *)(_aBuffer.contents)) = _typeA;
    
    return self;
}

- (void)configFunctionNameWithDimensionType:(DimensionType)dimensionType {    
    if (Dimension1D == _dimensionType) {
        _functionName = @BROU_METAL(PReLu1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(PReLu2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(PReLu3D);
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)configEncoderBufferWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                           inputBuffer:(id<MTLBuffer>)inputBuffer
                          outputBuffer:(id<MTLBuffer>)outputBuffer {
    [encoder setBuffer:inputBuffer  offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBuffer:_aBuffer     offset:0 atIndex:2];
    [encoder setBuffer:_shape       offset:0 atIndex:3];
}

@end

#endif
