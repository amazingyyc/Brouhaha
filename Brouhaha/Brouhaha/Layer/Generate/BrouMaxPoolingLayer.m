#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(MaxPoolingLayer)

- (void)configFunctionName {
    _name = @BROU_OBJECT_NAME(MaxPoolingLayer);
    
    _functionName = @BROU_METAL(MaxPooling);
}

@end

#endif
