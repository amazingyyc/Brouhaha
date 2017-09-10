#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(DilatedConvolutionLayer) : BROU_OBJECT(ConvolutionLayer) {
    int _dilatedY;
    int _dilatedX;
}

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
                      dilatedX:(int)dilatedX;

- (void)configParamsWithInputChannel:(int)inputChannel
                       outputChannel:(int)outputChannel
                        kernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX
                            dilatedY:(int)dilatedY
                            dilatedX:(int)dilatedX;

@end

#endif
