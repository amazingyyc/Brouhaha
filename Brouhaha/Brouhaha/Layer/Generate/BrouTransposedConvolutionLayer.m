#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@implementation BROU_OBJECT(TransposedConvolutionLayer)

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
                 originStrideX:(int)originStrideX {
    self = [super initWithName:@BROU_OBJECT_NAME(TransposedConvolutionLayer)];
    
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
    
    _functionName = @BROU_METAL(TransposedConvolution);
    
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
    
    [super configParamsWithInputChannel:_originOutputChannnel
                          outputChannel:_originInputChannel
                           kernelHeight:_originKernelHeight
                            kernelWidth:_originKernelWidth
                                 padTop:_originKernelHeight - _originPadTop  - 1
                                padLeft:_originKernelWidth  - _originPadLeft - 1
                                strideY:1
                                strideX:1];
}

- (void)configShapeWithDevice:(id<MTLDevice>)device {
    [super configShapeWithDevice:device];
    
    ConvolutionShape *convolutionShapeRef = (ConvolutionShape*)_convolutionShape.contents;
    
    convolutionShapeRef->insertY = _insertY;
    convolutionShapeRef->insertX = _insertX;
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    /**check params*/
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
    [encoder setBuffer:_kernel              offset:0 atIndex:1];
    [encoder setBuffer:_bias                offset:0 atIndex:2];
    [encoder setBuffer:output               offset:0 atIndex:3];
    [encoder setBuffer:_inputShape          offset:0 atIndex:4];
    [encoder setBuffer:_outputShape         offset:0 atIndex:5];
    [encoder setBuffer:_convolutionShape    offset:0 atIndex:6];
    
    /**
     * every thread will handle 4X4X4 output
     */
    MTLSize group = MTLSizeMake(8, 4, 1);
    MTLSize grid  = MTLSizeMake((outputShape.dim1 + 31) / 32,
                                (outputShape.dim0 + 15) / 16,
                                _outputChannelX4 / 4);
    
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
}

@end

#endif






