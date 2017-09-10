#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(PoolingLayer) : BrouLayer {
    int _kernelHeight;
    int _kernelWidth;
    
    int _padTop;
    int _padLeft;
    
    int _strideY;
    int _strideX;
    
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX;

- (void)configParamsWithKernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX;

- (void)configShapeWithDevice:(id<MTLDevice>)device;
- (void)configFunctionName;

@end

#endif
