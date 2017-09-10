#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(AddBiasLayer) : BrouLayer {
    /**the tensor shape*/
    DimensionType _dimensionType;
    
    id<MTLBuffer> _bias;
    id<MTLBuffer> _shape;
    
    int _biasLength;
    int _biasLengthX4;
}

/**
 * add a bias to a tensor, the tensor can be 1D, 2D, 3D
 * the bias will be added to the inner dimension
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                     floatBias:(void*)floatBias
                    biasLength:(int)biasLength
                 dimensionType:(DimensionType)dimensionType;

@end

#endif
