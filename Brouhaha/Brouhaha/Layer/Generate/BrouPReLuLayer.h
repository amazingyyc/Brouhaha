#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(PReLuLayer) : BROU_OBJECT(OperateLayer) {
    id<MTLBuffer> _aBuffer;
    
    type _typeA;
    float _floatA;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                             a:(float)a
                 dimensionType:(DimensionType)dimensionType;

@end

#endif
