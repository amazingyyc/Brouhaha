#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(TransposedConvolutionMMLayer)() {
    /**
     * the transposed convolution has a coperated convolution
     * the prefix _origin means the coperated convolution (origin convolution)
     * the property that without _origin mean the transpoed convolution's property
     */
    
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
     * insertX = strideX - 1
     * insertY = strideY - 1
     * insert 0-uints to the input on x/y axis
     */
    int _insertX;
    int _insertY;
    
    /**
     * _inputChannelX4 >= inputchannel and timed by 4
     * _inputChannelX4 >= outputChannel and timed by 4
     */
    int _inputChannelX4;
    int _outputChannelX4;
    
    /**if the convolution has a bias*/
    bool _haveBias;
    
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
    
    /**store the kernel and bias*/
    id<MTLBuffer> _kernel;
    id<MTLBuffer> _bias;
    
    /**store the params and dimension of input/output*/
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
    
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

@implementation BROU_OBJECT(TransposedConvolutionMMLayer)

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
                 originStrideX:(int)originStrideX {
    self = [super initWithName:@BROU_OBJECT_NAME(TransposedConvolutionMMLayer)];

    if (!self) {
        return self;
    }
    
    [self configParamsWithOriginInputChannel:originInputChannel
                         originOutputChannel:originOutputChannel
                          originKernelHeight:originKernelHeight
                           originKernelWidth:originKernelWidth
                                originPadTop:originPadTop
                               originPadLeft:originPadLeft
                               originStrideY:originStrideY
                               originStrideX:originStrideX];
    
    [self configBufferWithDevice:device floatKernel:floatKernel];
    [self configBufferWithDevice:device floatBias:floatBias];
    [self configShapeWithDevice:device];
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configParamsWithOriginInputChannel:(int)originInputChannel
                       originOutputChannel:(int)originOutputChannel
                        originKernelHeight:(int)originKernelHeight
                         originKernelWidth:(int)originKernelWidth
                              originPadTop:(int)originPadTop
                             originPadLeft:(int)originPadLeft
                             originStrideY:(int)originStrideY
                             originStrideX:(int)originStrideX {
    NSAssert(originInputChannel  > 0, @"the originInputChannel must > 0");
    NSAssert(originOutputChannel > 0, @"the originOutputChannel must > 0");
    NSAssert(originKernelHeight  > 0, @"the originKernelHeight must > 0");
    NSAssert(originKernelWidth   > 0, @"the originKernelWidth must > 0");
    NSAssert(originPadTop  >= 0, @"the originPadTop must >= 0");
    NSAssert(originPadLeft >= 0, @"the originPadLeft must >= 0");
    NSAssert(originStrideY  > 0, @"the originStrideY must > 0");
    NSAssert(originStrideX  > 0, @"the originStrideX must > 0");
    NSAssert(originKernelHeight > originPadTop, @"the originKernelHeight must > originPadTop");
    NSAssert(originKernelWidth  > originPadLeft, @"the originKernelWidth must > originPadLeft");
    
    _originInputChannel     = originInputChannel;
    _originOutputChannnel   = originOutputChannel;
    _originKernelHeight     = originKernelHeight;
    _originKernelWidth      = originKernelWidth;
    _originPadTop           = originPadTop;
    _originPadLeft          = originPadLeft;
    _originStrideY          = originStrideY;
    _originStrideX          = originStrideX;
    
    _insertX = _originStrideX - 1;
    _insertY = _originStrideY - 1;
    
    _inputChannel  = _originOutputChannnel;
    _outputChannel = _originInputChannel;
    _kernelHeight  = _originKernelHeight;
    _kernelWidth   = _originKernelWidth;
    
    _padTop  = _originKernelHeight - _originPadTop  - 1;
    _padLeft = _originKernelWidth  - _originPadLeft - 1;
    _strideY = 1;
    _strideX = 1;
    
    _inputChannelX4  = (_inputChannel  + 3) / 4 * 4;
    _outputChannelX4 = (_outputChannel + 3) / 4 * 4;
    
    _inputMatrixRow  = _inputChannelX4;
    
    _mediateMatrixRow = _outputChannelX4 * _kernelHeight * _kernelWidth;
    
    _kernelMatrixRow = _inputChannelX4;
    _kernelMatrixCol = _outputChannelX4 * _kernelHeight * _kernelWidth;
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatKernel:(void*)floatKernel {
    void *realKernel = NULL;
    
#if defined(real_is_half)
    realKernel = malloc(sizeof(type) * _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
    
    convertFloat32ToFloat16(floatKernel, realKernel, _outputChannel * _kernelHeight * _kernelWidth * _inputChannel);
#elif defined(real_is_float)
    realKernel = floatKernel;
#endif
    
    if (@available(iOS 9.0, *)) {
        _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _kernel = [device newBufferWithLength:sizeof(type) * _outputChannelX4 * _kernelHeight * _kernelWidth * _inputChannelX4
                                      options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    BROU(TransposeMatrix)(realKernel,
                          _outputChannel * _kernelHeight * _kernelWidth,
                          _inputChannel,
                          _kernel.contents,
                          _kernelMatrixRow,
                          _kernelMatrixCol);
    
#if defined(real_is_half)
    free(realKernel);
#endif
}

- (void)configBufferWithDevice:(id<MTLDevice>)device floatBias:(void*)floatBias {
    if (NULL == floatBias) {
        _haveBias = false;
        
        /**if the MTLBuffer is nil or length is 0, the Metal will be crash, don't why*/
        if (@available(iOS 9.0, *)) {
            _bias = [device newBufferWithLength:sizeof(type)
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        } else {
            _bias = [device newBufferWithLength:sizeof(type)
                                        options:MTLResourceCPUCacheModeDefaultCache];
        }
        
        return;
    }
    
    void *realBias = NULL;
    
#if defined(real_is_half)
    /**the real is half */
    realBias = malloc(sizeof(type) * _outputChannel);
    
    convertFloat32ToFloat16(floatBias,
                            realBias,
                            _outputChannel);
#elif defined(real_is_float)
    /**the real is float*/
    realBias = floatBias;
#endif
    
    if (@available(iOS 9.0, *)) {
        _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _bias = [device newBufferWithLength:sizeof(type) * _outputChannelX4
                                    options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    memcpy(_bias.contents, realBias, sizeof(type) * _outputChannel);
    
#if defined(real_is_half)
    free(realBias);
#endif
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    if (@available(iOS 9.0, *)) {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _input2MatrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _matrixMultipyShape = [device newBufferWithLength:sizeof(TensorShape)
                                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        
        _matrix2OutputShape = [device newBufferWithLength:sizeof(TensorShape)
                                                  options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache];
        
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache];
        
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache];
        
        _input2MatrixShape = [device newBufferWithLength:sizeof(TensorShape)
                                                 options:MTLResourceCPUCacheModeDefaultCache];
        
        _matrixMultipyShape = [device newBufferWithLength:sizeof(TensorShape)
                                                  options:MTLResourceCPUCacheModeDefaultCache];
        
        _matrix2OutputShape = [device newBufferWithLength:sizeof(TensorShape)
                                                  options:MTLResourceCPUCacheModeDefaultCache];
    }

    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    convolutionShapeRef->padTop       = _padTop;
    convolutionShapeRef->padLeft      = _padLeft;
    convolutionShapeRef->strideY      = _strideY;
    convolutionShapeRef->strideX      = _strideX;
    convolutionShapeRef->haveBias     = _haveBias;
    convolutionShapeRef->insertY      = _insertY;
    convolutionShapeRef->insertX      = _insertX;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    _input2MatrixFunctionName   = @BROU_METAL(TransposedConvolutionInput2Matrix);
    _matrixMultiplyFunctionName = @BROU_METAL(MatrixMultiply);
    _matrix2OutputFunctionName  = @BROU_METAL(TransposedConvolutionMatrix2Output);
    
    /**get the function*/
    NSError *error = nil;
    
    /**input 2 matrix*/
    id<MTLFunction> input2MatrixFunction = [library newFunctionWithName:_input2MatrixFunctionName];
    
    NSAssert(input2MatrixFunction, @"init input2Matrix function error!");
    
    _input2MatrixComputePipelineState = [device newComputePipelineStateWithFunction:input2MatrixFunction
                                                                              error:&error];
    
    NSAssert(_input2MatrixComputePipelineState, @"input2Matrix function error!");
    
    /**matrix multiply*/
    id<MTLFunction> matrixMultiplyFunction = [library newFunctionWithName:_matrixMultiplyFunctionName];
    
    NSAssert(matrixMultiplyFunction, @"init matrix multiply function error!");
    
    _matrixMultiplyComputePipelineState = [device newComputePipelineStateWithFunction:matrixMultiplyFunction
                                                                                error:&error];
    
    NSAssert(_matrixMultiplyComputePipelineState, @"init matrix multiply function error!");
    
    /**matrix to output*/
    id<MTLFunction> matrix2OutputFunction = [library newFunctionWithName:_matrix2OutputFunctionName];
    
    NSAssert(matrix2OutputFunction, @"init matrix 2 output function error!");
    
    _matrix2OutputComputePipelineState = [device newComputePipelineStateWithFunction:matrix2OutputFunction
                                                                               error:&error];
    
    NSAssert(_matrix2OutputComputePipelineState, @"init matrix 2 output function error!");
}

- (void)checkParamsWithInput:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    NSAssert(3 == input.dimension, @"The input tensor's dimension must be 3");
    NSAssert(input.height > 0 && input.width > 0 && input.channel > 0, @"the dim of input must > 0");
    NSAssert(_inputChannelX4  == input.innermostDimX4, @"the input shape is error!");
    
    NSAssert(3 == output.dimension, @"The output tensor's dimension must be 3");
    NSAssert(output.height > 0 && output.width > 0 && output.channel > 0, @"the dim of output must > 0");
    NSAssert(_outputChannelX4  == output.innermostDimX4, @"the output shape is error!");
}

- (void)configMetalShapeWithInput:(id<BrouTensor>)input output:(id<BrouTensor>)output {
    _inputMatrixRow = _inputChannelX4;
    _inputMatrixCol = (input.dim1 * input.dim0 + 3) / 4 * 4;
    
    _mediateMatrixRow = _outputChannelX4 * _kernelHeight * _kernelWidth;
    _mediateMatrixCol = _inputMatrixCol;
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = input.dim0;
    inputShapeRef->dim1 = input.dim1;
    inputShapeRef->dim2 = input.innermostDimX4;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = output.dim0;
    outputShapeRef->dim1 = output.dim1;
    outputShapeRef->dim2 = output.innermostDimX4;
    
    TensorShape *input2MatrixShapeRef = (TensorShape*)_input2MatrixShape.contents;
    input2MatrixShapeRef->dim0 = input.dim1 * input.dim0;
    input2MatrixShapeRef->dim1 = _inputMatrixRow;
    input2MatrixShapeRef->dim2 = _inputMatrixCol;
    
    TensorShape *matrixMultipyShapeRef = (TensorShape*)_matrixMultipyShape.contents;
    matrixMultipyShapeRef->dim0  = _mediateMatrixRow;
    matrixMultipyShapeRef->dim1  = _kernelMatrixRow;
    matrixMultipyShapeRef->dim2  = _mediateMatrixCol;
    
    TensorShape *matrix2OutputShapeRef = (TensorShape*)_matrix2OutputShape.contents;
    matrix2OutputShapeRef->dim0 = _mediateMatrixRow;
    matrix2OutputShapeRef->dim1 = _mediateMatrixCol;
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput:input output:output];
    [self configMetalShapeWithInput:input output:output];
    
    /**malloc memory*/
    if (!_inputMatrix || _inputMatrix.length < sizeof(type) * _inputMatrixRow * _inputMatrixCol) {
        if (@available(iOS 9.0, *)) {
            _inputMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _inputMatrixRow * _inputMatrixCol
                                                             options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        } else {
            _inputMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _inputMatrixRow * _inputMatrixCol
                                                             options:MTLResourceCPUCacheModeDefaultCache];
        }
    }
    
    if (_mediateMatrix || _mediateMatrix.length < sizeof(type) * _mediateMatrixRow * _mediateMatrixCol) {
        if (@available(iOS 9.0, *)) {
            _mediateMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _mediateMatrixRow * _mediateMatrixCol
                                                               options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        } else {
            _mediateMatrix = [commandBuffer.device newBufferWithLength:sizeof(type) * _mediateMatrixRow * _mediateMatrixCol
                                                               options:MTLResourceCPUCacheModeDefaultCache];
        }
    }
    
    id<MTLComputeCommandEncoder> input2MatrixEncoder = [commandBuffer computeCommandEncoder];
    [input2MatrixEncoder setComputePipelineState:_input2MatrixComputePipelineState];
    [input2MatrixEncoder setBuffer:input.tensorBuffer offset:0 atIndex:0];
    [input2MatrixEncoder setBuffer:_inputMatrix       offset:0 atIndex:1];
    [input2MatrixEncoder setBuffer:_input2MatrixShape offset:0 atIndex:2];
    
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((_inputMatrixCol + 31) / 32,
                                (_inputMatrixRow + 15) / 16,
                                1);
    
    [input2MatrixEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [input2MatrixEncoder endEncoding];
    
    id<MTLComputeCommandEncoder> matrixMultipyEncoder = [commandBuffer computeCommandEncoder];
    [matrixMultipyEncoder setComputePipelineState:_matrixMultiplyComputePipelineState];
    [matrixMultipyEncoder setBuffer:_kernel             offset:0 atIndex:0];
    [matrixMultipyEncoder setBuffer:_inputMatrix        offset:0 atIndex:1];
    [matrixMultipyEncoder setBuffer:_mediateMatrix      offset:0 atIndex:2];
    [matrixMultipyEncoder setBuffer:_matrixMultipyShape offset:0 atIndex:3];
    
    
    group = MTLSizeMake(8, 4, 1);
    grid   = MTLSizeMake((_mediateMatrixCol + 31) / 32,
                         (_mediateMatrixRow + 15) / 16,
                         1);
    [matrixMultipyEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [matrixMultipyEncoder endEncoding];
    
    id<MTLComputeCommandEncoder> matrix2OutputEncoder = [commandBuffer computeCommandEncoder];
    [matrix2OutputEncoder setComputePipelineState:_matrix2OutputComputePipelineState];
    [matrix2OutputEncoder setBuffer:_mediateMatrix         offset:0 atIndex:0];
    [matrix2OutputEncoder setBuffer:output.tensorBuffer    offset:0 atIndex:1];
    [matrix2OutputEncoder setBuffer:_bias                  offset:0 atIndex:2];
    [matrix2OutputEncoder setBuffer:_matrix2OutputShape    offset:0 atIndex:3];
    [matrix2OutputEncoder setBuffer:_inputShape            offset:0 atIndex:4];
    [matrix2OutputEncoder setBuffer:_outputShape           offset:0 atIndex:5];
    [matrix2OutputEncoder setBuffer:_convolutionShape      offset:0 atIndex:6];
    
    group = MTLSizeMake(8, 4, 1);
    grid   = MTLSizeMake((output.dim1 + 31) / 32,
                         (output.dim0 + 15) / 16,
                          output.innermostDimX4);
    
    [matrix2OutputEncoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [matrix2OutputEncoder endEncoding];
}

@end

#endif









