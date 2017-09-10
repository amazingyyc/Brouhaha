#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(TanHLayer)

- (void)configFunctionNameWithDimensionType:(DimensionType)dimensionType {
    _name = @BROU_OBJECT_NAME(TanHLayer);
    
    if (Dimension1D == _dimensionType) {
        _functionName = @BROU_METAL(TanH1D);
    } else if (Dimension2D == _dimensionType) {
        _functionName = @BROU_METAL(TanH2D);
    } else if (Dimension3D == _dimensionType) {
        _functionName = @BROU_METAL(TanH3D);
    } else {
        NSAssert(false, @"the dimension type is error");
    }
}

- (void)configEncoderBufferWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                           inputBuffer:(id<MTLBuffer>)inputBuffer
                          outputBuffer:(id<MTLBuffer>)outputBuffer {
    [encoder setBuffer:inputBuffer  offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    [encoder setBuffer:_shape       offset:0 atIndex:2];
}

@end

#endif
