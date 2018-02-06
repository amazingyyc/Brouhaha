#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(TransposedConvolutionLayer) : BrouLayer

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
