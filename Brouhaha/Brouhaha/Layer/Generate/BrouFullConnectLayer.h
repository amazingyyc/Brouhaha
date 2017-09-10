#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(FullConnectLayer) : BrouLayer {
    /**the input features channel*/
    int _inputChannel;
    
    /**the output feature channel*/
    int _outputChannel;
    
    /**
     * _inputChannelX4 >= inputchannel and timed by 4
     * _inputChannelX4 >= outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;
    
    /**if the convolution has a bias*/
    bool _haveBias;
    
    /**store the kernel and bias*/
    id<MTLBuffer> _weigths;
    id<MTLBuffer> _bias;
    
    id<MTLBuffer> _shape;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  floatWeights:(void*)floatWeight
                     floatBias:(void*)floatBias
                 intputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel;

@end

#endif











