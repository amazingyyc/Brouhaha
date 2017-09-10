#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(LinearLayer) : BROU_OBJECT(OperateLayer) {
    id<MTLBuffer> _abBuffer;
    
    type _typeA;
    type _typeB;
    
    float _floatA;
    float _floatB;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                             a:(float)a
                             b:(float)b
                 dimensionType:(DimensionType)dimensionType;

@end

#endif
