#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(ConvolutionLayer) : BrouLayer {
    /**the input features channel*/
    int _inputChannel;
    
    /**the output feature channel*/
    int _outputChannel;
    
    /**the kernel dimesnion is (outputChannel, kernelHeight, kernelWidth, inputChannel)*/
    int _kernelHeight;
    int _kernelWidth;
    
    /**the pad*/
    int _padLeft;
    int _padTop;
    
    /**stride of kernel*/
    int _strideX;
    int _strideY;
    
    /**
     * _inputChannelX4 >= inputchannel and timed by 4
     * _inputChannelX4 >= outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;
    
    /**if the convolution has a bias*/
    bool _haveBias;
    
    /**store the kernel and bias*/
    id<MTLBuffer> _kernel;
    id<MTLBuffer> _bias;
    
    /**store the params and dimension of input/output*/
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
}

/**
 * this init function can init float32 and float16 layer
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   floatKernel:(void*)floatKernel
                     floatBias:(void*)floatBias
                  inputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX;

- (void)configBufferWithDevice:(id<MTLDevice>)device floatKernel:(void*)floatKernel;
- (void)configBufferWithDevice:(id<MTLDevice>)device floatBias:(void*)floatBias;

- (void)configParamsWithInputChannel:(int)inputChannel
                       outputChannel:(int)outputChannel
                        kernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX;

- (void)configShapeWithDevice:(id<MTLDevice>)device;

- (void)checkParamsWithInputShape:(TensorShape)inputShape
                      outputShape:(TensorShape)outputShape;

@end

#endif














