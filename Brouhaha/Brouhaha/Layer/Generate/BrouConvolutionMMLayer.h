#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(ConvolutionMMLayer) : BROU_OBJECT(ConvolutionLayer) {
    /**
     * _kernelHeightXkernelWidthXinputChannelX4 = kernelHeight * kernelWidth * inputChannel4
     */
    int _kernelHeightXkernelWidthXinputChannelX4;
    
    /**store the matrix multipy shape*/
    id<MTLBuffer> _matrixShape;
    
    /**stror the input matrix*/
    id<MTLBuffer> _inputMatrix;
    
    /**the fucntion name of onvert input to a matrix*/
    NSString *_convertInputFunctionName;
    
    /**the pipe state*/
    id<MTLComputePipelineState> _convertInputComputePipelineState;
}

@end

#endif
