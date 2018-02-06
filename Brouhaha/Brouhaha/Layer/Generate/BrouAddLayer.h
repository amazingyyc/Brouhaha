#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(AddLayer) : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType;

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                      input1:(id<BrouTensor>)input1
                      input2:(id<BrouTensor>)input2
                      output:(id<BrouTensor>)output;

@end

#endif
