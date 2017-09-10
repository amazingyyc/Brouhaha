#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(TransposedConvolutionMMLayer) : BROU_OBJECT(TransposedConvolutionLayer) {
    /**
     * _inputMatrixRow = inputChannelx4
     * _inputMatrixCol = [inputHeight * inputWidth]X4
     */
    int _inputMatrixRow;
    int _inputMatrixCol;
    
    /**
     * _mediateMatrixRow = _outputChannelX4 * kernelHeight * kernelWidth
     * _mediateMatrixCol = _inputMatrixCol = [inputHeight * inputWidth]X4
     */
    int _mediateMatrixRow;
    int _mediateMatrixCol;
    
    /**
     * _kernelMatrixRow = inputChannelx4
     * _kernelMatrixCol = _outputChannelX4 * kernelHeight * kernelWidth
     */
    int _kernelMatrixRow;
    int _kernelMatrixCol;
    
    id<MTLBuffer> _input2MatrixShape;
    id<MTLBuffer> _matrixMultipyShape;
    id<MTLBuffer> _matrix2OutputShape;
    
    id<MTLBuffer> _inputMatrix;
    id<MTLBuffer> _mediateMatrix;
    
    NSString *_input2MatrixFunctionName;
    NSString *_matrixMultiplyFunctionName;
    NSString *_matrix2OutputFunctionName;
    
    id<MTLComputePipelineState> _input2MatrixComputePipelineState;
    id<MTLComputePipelineState> _matrixMultiplyComputePipelineState;
    id<MTLComputePipelineState> _matrix2OutputComputePipelineState;
}

@end

#endif
