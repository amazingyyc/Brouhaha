#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(TransposedConvolutionLayer) : BROU_OBJECT(ConvolutionLayer) {
    /**
     * the transposed convolution has a coperated convolution
     * the prefix _origin means the coperated convolution (origin convolution)
     * the property that without _origin mean the transpoed convolution's property
     */
    
    /**
     * the origin convolution input dimension
     */
    int _originInputChannel;
    
    /**
     * the origin convolution output dimension
     */
    int _originOutputChannnel;
    
    /**
     * the origin convoluton's kernel
     */
    int _originKernelHeight;
    int _originKernelWidth;
    
    /**
     * the origin pad
     */
    int _originPadLeft;
    int _originPadTop;
    
    /**
     * the origin stride
     */
    int _originStrideX;
    int _originStrideY;
    
    /**
     * insertX = strideX - 1
     * insertY = strideY - 1
     * insert 0-uints to the input on x/y axis
     */
    int _insertX;
    int _insertY;
}

/**
 * this init function can init float32 and float16 layer
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   floatKernel:(void*)floatKernel
                     floatBias:(void*)floatBias
            originInputChannel:(int)originInputChannel
           originOutputChannel:(int)originOutputChannel
            originKernelHeight:(int)originKernelHeight
             originKernelWidth:(int)originKernelWidth
                  originPadTop:(int)originPadTop
                 originPadLeft:(int)originPadLeft
                 originStrideY:(int)originStrideY
                 originStrideX:(int)originStrideX;

@end

#endif
