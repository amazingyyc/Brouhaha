#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(MaxPoolingLayer) () {
    int _kernelHeight;
    int _kernelWidth;
    
    int _padTop;
    int _padLeft;
    
    int _strideY;
    int _strideX;
    
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    id<MTLBuffer> _convolutionShape;
    
    NSString *_functionName;
    
    /**the Metal computePipelineState*/
    id<MTLComputePipelineState> _computePipelineState;
}

@end

@implementation BROU_OBJECT(MaxPoolingLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX {
    self = [super initWithName:@BROU_OBJECT_NAME(MaxPoolingLayer)];
    
    if (!self) {
        return self;
    }
    
    [self configParamsWithKernelHeight:kernelHeight
                           kernelWidth:kernelWidth
                                padTop:padTop
                               padLeft:padLeft
                               strideY:strideY
                               strideX:strideX];
    
    [self configShapeWithDevice:device];
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    if (@available(iOS 9.0, *)) {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                          options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    if (@available(iOS 9.0, *)) {
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                           options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    if (@available(iOS 9.0, *)) {
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    } else {
        _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                                options:MTLResourceCPUCacheModeDefaultCache];
    }
    
    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    convolutionShapeRef->padTop       = _padTop;
    convolutionShapeRef->padLeft      = _padLeft;
    convolutionShapeRef->strideY      = _strideY;
    convolutionShapeRef->strideX      = _strideX;
}

- (void)configParamsWithKernelHeight:(int)kernelHeight
                         kernelWidth:(int)kernelWidth
                              padTop:(int)padTop
                             padLeft:(int)padLeft
                             strideY:(int)strideY
                             strideX:(int)strideX {
    NSAssert(kernelHeight  > 0, @"the kernelHeight must > 0");
    NSAssert(kernelWidth   > 0, @"the kernelWidth must > 0");
    NSAssert(padTop  >= 0, @"the padTop must >= 0");
    NSAssert(padLeft >= 0, @"the padLeft must >= 0");
    NSAssert(strideY  > 0, @"the strideY must > 0");
    NSAssert(strideX  > 0, @"the padLeft must > 0");
    
    _kernelHeight = kernelHeight;
    _kernelWidth  = kernelWidth;
    
    _padTop  = padTop;
    _padLeft = padLeft;
    
    _strideY = strideY;
    _strideX = strideX;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    /**init function name*/
    _functionName = @BROU_METAL(MaxPooling);
    
    id<MTLFunction> function = [library newFunctionWithName:_functionName];
    
    NSAssert(function, @"init %@ function:%@ error!", self.name, _functionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    
    NSAssert(_computePipelineState, @"init %@ ComputePipelineState error:%@", self.name, error);
}

- (void)checkParamsWithInput:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    NSAssert(3 == input.dimension, @"The input tensor's dimension must be 3");
    NSAssert(input.height > 0, @"the input height must > 0");
    NSAssert(input.width > 0, @"the input width must > 0");
    
    NSAssert(3 == output.dimension, @"The output tensor's dimension must be 3");
    NSAssert(output.height > 0, @"the output height must > 0");
    NSAssert(output.width > 0, @"the output width must > 0");
    
    NSAssert(input.channel ==  output.channel, @"the input channel must equal to output channel");
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput:input output:output];
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = input.dim0;
    inputShapeRef->dim1 = input.dim1;
    inputShapeRef->dim2 = input.innermostDimX4;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = output.dim0;
    outputShapeRef->dim1 = output.dim1;
    outputShapeRef->dim2 = output.innermostDimX4;
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input.tensorBuffer   offset:0 atIndex:0];
    [encoder setBuffer:output.tensorBuffer  offset:0 atIndex:1];
    [encoder setBuffer:_inputShape          offset:0 atIndex:2];
    [encoder setBuffer:_outputShape         offset:0 atIndex:3];
    [encoder setBuffer:_convolutionShape    offset:0 atIndex:4];
    
    /**
     * every thread will handle 1X1X4 output
     */
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((output.dim1 + 7) / 8,
                                (output.dim0 + 3) / 4,
                                 output.innermostDimX4 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif
