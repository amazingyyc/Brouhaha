#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(LinearLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                             a:(float)a
                             b:(float)b
                 dimensionType:(DimensionType)dimensionType {
    self = [super initWithDevice:device library:library dimensionType:dimensionType];
    
    if (!self) {
        return self;
    }
    
    _name = @BROU_OBJECT_NAME(LinearLayer);
    
    _abBuffer = [device newBufferWithLength:sizeof(type) * 2
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _floatA = a;
    _floatB = b;
    
#if defined(real_is_half)
    _typeA = convertFloat32ToFloat16OneNumber((uint32_t*)(&a));
    _typeB = convertFloat32ToFloat16OneNumber((uint32_t*)(&b));
#elif defined(real_is_float)
    _typeA = a;
    _typeB = b;
#endif
    
    type *typeRef = (type *)_abBuffer.contents;
    typeRef[0] = _typeA;
    typeRef[1] = _typeB;
    
    return self;
}

- (void)configFunctionNameWithDimensionType:(DimensionType)dimensionType {
    if (Dimension1D == _dimensionType) {
        _functionName = @BROU_METAL(Linear1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(Linear2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(Linear3D);
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)configEncoderBufferWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                           inputBuffer:(id<MTLBuffer>)inputBuffer
                          outputBuffer:(id<MTLBuffer>)outputBuffer {
    [encoder setBuffer:inputBuffer  offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBuffer:_abBuffer    offset:0 atIndex:2];
    [encoder setBuffer:_shape       offset:0 atIndex:3];
}

@end

#endif









