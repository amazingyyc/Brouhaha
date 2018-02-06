#if defined(from) && defined(to) && defined(BROU_CONVERT_OBJECT) && defined(BROU_CONVERT_METAL)

@interface BROU_CONVERT_OBJECT(from, to) : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType;

@end

#endif
