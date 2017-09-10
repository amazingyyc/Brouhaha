#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(AddLayer) : BrouLayer {
    DimensionType _dimensionType;
    id<MTLBuffer> _shape;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType;

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                          input1:(id<MTLBuffer>)input1
                          input2:(id<MTLBuffer>)input2
                          output:(id<MTLBuffer>)output
                           shape:(TensorShape)shape;

@end

#endif
