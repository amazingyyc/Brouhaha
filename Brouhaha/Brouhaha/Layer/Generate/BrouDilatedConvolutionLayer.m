#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(DilatedConvolutionLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   floatKernel:(void *)floatKernel
                     floatBias:(void *)floatBias
                  inputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX
                       dilateY:(int)dilatedY
                      dilatedX:(int)dilatedX {
    self = [super initWithName:@BROU_OBJECT_NAME(DilatedConvolutionLayer)];
    
    if (!self) {
        return self;
    }
    
    [self configParamsWithInputChannel:inputChannel
                         outputChannel:outputChannel
                          kernelHeight:kernelHeight
                           kernelWidth:kernelWidth
                                padTop:padTop
                               padLeft:padLeft
                               strideY:strideY
                               strideX:strideX
                              dilatedY:dilatedY
                              dilatedX:dilatedX];
    
    [self configBufferWithDevice:device floatKernel:floatKernel];
    [self configBufferWithDevice:device floatBias:floatBias];
    [self configShapeWithDevice:device];
    
    /**init function name*/
    _functionName = @BROU_METAL(DilatedConvolution);
    
    /**config Metal function*/
    [self configComputePipelinesStateWithDevice:device
                                        library:library];
    
    return self;
}

- (void)configParamsWithInputChannel:(int)inputChannel
                       outputChannel:(int)outputChannel
                        kernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX
                            dilatedY:(int)dilatedY
                            dilatedX:(int)dilatedX {
    [super configParamsWithInputChannel:inputChannel
                          outputChannel:outputChannel
                           kernelHeight:kernelHeight
                            kernelWidth:kernelWidth
                                 padTop:padTop
                                padLeft:padLeft
                                strideY:strideY
                                strideX:strideX];
    
    NSAssert(dilatedY > 0, @"the dilatedY must > 0");
    NSAssert(dilatedX > 0, @"the dilatedX must > 0");
    
    _dilatedY = dilatedY;
    _dilatedX = dilatedX;
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    [super configShapeWithDevice:device];
    
    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    
    convolutionShapeRef->dilatedY = _dilatedY;
    convolutionShapeRef->dilatedX = _dilatedX;
}

@end

#endif







