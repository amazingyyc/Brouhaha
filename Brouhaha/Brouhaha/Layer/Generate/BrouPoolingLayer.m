#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(PoolingLayer)

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  kernelHeight:(int)kernelHeight
                   kernelWidth:(int)kernelWidth
                        padTop:(int)padTop
                       padLeft:(int)padLeft
                       strideY:(int)strideY
                       strideX:(int)strideX {
    self = [super initWithName:@BROU_OBJECT_NAME(PoolingLayer)];
    
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
    [self configFunctionName];
    [self configComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

- (void)configFunctionName {
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

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    _inputShape = [device newBufferWithLength:sizeof(TensorShape)
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _outputShape = [device newBufferWithLength:sizeof(TensorShape)
                                       options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _convolutionShape = [device newBufferWithLength:sizeof(ConvolutionShape)
                                            options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    convolutionShapeRef->kernelHeight = _kernelHeight;
    convolutionShapeRef->kernelWidth  = _kernelWidth;
    convolutionShapeRef->padTop       = _padTop;
    convolutionShapeRef->padLeft      = _padLeft;
    convolutionShapeRef->strideY      = _strideY;
    convolutionShapeRef->strideX      = _strideX;
}

- (void)checkParamsWithInputShape:(TensorShape)inputShape
                      outputShape:(TensorShape)outputShape {
    NSAssert(inputShape.dim0 > 0, @"the input height must > 0");
    NSAssert(inputShape.dim1 > 0, @"the input width must > 0");
    NSAssert(outputShape.dim0 > 0, @"the output height must > 0");
    NSAssert(outputShape.dim1 > 0, @"the output width must > 0");
    NSAssert(inputShape.dim2 ==  outputShape.dim2 && 0 == inputShape.dim2 % 4,
             @"the channel must be timed by 4 and inputChannel must equal to outputChannel");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    [self checkParamsWithInputShape:inputShape outputShape:outputShape];
    
    TensorShape *inputShapeRef = (TensorShape*)_inputShape.contents;
    inputShapeRef->dim0 = inputShape.dim0;
    inputShapeRef->dim1 = inputShape.dim1;
    inputShapeRef->dim2 = inputShape.dim2;
    
    TensorShape *outputShapeRef = (TensorShape*)_outputShape.contents;
    outputShapeRef->dim0 = outputShape.dim0;
    outputShapeRef->dim1 = outputShape.dim1;
    outputShapeRef->dim2 = outputShape.dim2;
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input                offset:0 atIndex:0];
    [encoder setBuffer:output               offset:0 atIndex:1];
    [encoder setBuffer:_inputShape          offset:0 atIndex:2];
    [encoder setBuffer:_outputShape         offset:0 atIndex:3];
    [encoder setBuffer:_convolutionShape    offset:0 atIndex:4];
    
    /**
     * every thread will handle 1X1X4 output
     */
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((outputShape.dim1 + 7) / 8,
                                (outputShape.dim0 + 3) / 4,
                                 outputShape.dim2 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif











