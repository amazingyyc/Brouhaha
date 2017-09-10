#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

/**
 * this a base Object of Tanh ReLu PReLU Linear layer
 */
@interface BROU_OBJECT(OperateLayer) : BrouLayer {
    DimensionType _dimensionType;
    
    id<MTLBuffer> _shape;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType;

- (void)configFunctionNameWithDimensionType:(DimensionType)dimensionType;

- (void)configEncoderBufferWithEncoder:(id<MTLComputeCommandEncoder>)encoder
                           inputBuffer:(id<MTLBuffer>)inputBuffer
                          outputBuffer:(id<MTLBuffer>)outputBuffer;

@end

#endif
