#if defined(from) && defined(to) && defined(BROU_CONVERT_OBJECT) && defined(BROU_CONVERT_METAL)

@interface BROU_CONVERT_OBJECT(from, to) : BrouLayer {
    DimensionType _dimensionType;
    
    id<MTLBuffer> _shape;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                 dimensionType:(DimensionType)dimensionType;

@end

#endif
