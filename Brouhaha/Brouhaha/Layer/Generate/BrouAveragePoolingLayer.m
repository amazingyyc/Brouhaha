#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(AveragePoolingLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX
                       withPad:(BOOL)withPad {
    self = [super initWithName:@BROU_OBJECT_NAME(AveragePoolingLayer)];
    
    if (!self) {
        return self;
    }
    
    [self configParamsWithKernelHeight:kernelHeight
                           kernelWidth:kernelWidth
                                padTop:padTop
                               padLeft:padLeft
                               strideY:strideY
                               strideX:strideX];

    _withPad = withPad;
    
    if (_withPad) {
        _functionName = @BROU_METAL(AveragePooling);
    } else {
        _functionName = @BROU_METAL(AveragePoolingWithoutPad);
    }
    
    [self configShapeWithDevice:device];
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

@end

#endif
