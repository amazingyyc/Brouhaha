#import "BrouResidualLayer_half.h"

@implementation BrouResidualLayer_half

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  floatWeight1:(void*)floatWeight1
                  floatWeight2:(void*)floatWeight2
                   floatAlpha1:(void*)floatAlpha1
                    floatBeta1:(void*)floatBeta1
                   floatAlpha2:(void*)floatAlpha2
                    floatBeta2:(void*)floatBeta2
                       channel:(int)channel {
    self = [super initWithName:@"BrouResidualLayer_half"];
    
    if (!self) {
        return self;
    }
    
    _channel   = channel;
    _channelX4 = (_channel + 3) / 4 * 4;
    
    _conv1 = [[BrouConvolutionMMLayer_half alloc] initWithDevice:device
                                                          library:library
                                                      floatKernel:floatWeight1
                                                        floatBias:nil
                                                     inputChannel:_channel
                                                    outputChannel:_channel
                                                     kernelHeight:3
                                                      kernelWidth:3
                                                           padTop:1
                                                          padLeft:1
                                                          strideY:1
                                                          strideX:1];
    
    _batchNorm1 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                    library:library
                                                                    epsilon:0.001
                                                                 floatAlpha:floatAlpha1
                                                                  floatBeta:floatBeta1
                                                                    channel:_channel];
    
    _relu1 = [[BrouReLuLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    _conv2 = [[BrouConvolutionMMLayer_half alloc] initWithDevice:device
                                                          library:library
                                                      floatKernel:floatWeight2
                                                        floatBias:nil
                                                     inputChannel:_channel
                                                    outputChannel:_channel
                                                     kernelHeight:3
                                                      kernelWidth:3
                                                           padTop:1
                                                          padLeft:1
                                                          strideY:1
                                                          strideX:1];
    
    _batchNorm2 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                    library:library
                                                                    epsilon:0.001
                                                                 floatAlpha:floatAlpha2
                                                                  floatBeta:floatBeta2
                                                                    channel:_channel];
    
    _add = [[BrouAddLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    return self;
}

- (void)checkParamsWithInputShape:(TensorShape)inputShape
                      outputShape:(TensorShape)outputShape {
    NSAssert(inputShape.dim0 == outputShape.dim0 && inputShape.dim0 > 0 &&
             inputShape.dim1 == outputShape.dim1 && inputShape.dim1 > 0 &&
             inputShape.dim2 == outputShape.dim2 && inputShape.dim2 > 0,
             @"the shape is error!");
    
    NSAssert(inputShape.dim2 == _channelX4, @"the shape is error!");
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    [self checkParamsWithInputShape:inputShape outputShape:outputShape];
    
    int length = sizeof(uint16_t)*inputShape.dim0 * inputShape.dim1 * inputShape.dim2;
    
    if (!_buffer1 || _buffer1.length < length) {
        _buffer1 = [commandBuffer.device newBufferWithLength:length
                                                     options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    }
    
    if (!_buffer2 || _buffer2.length < length) {
        _buffer2 = [commandBuffer.device newBufferWithLength:length
                                                     options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    }
    
    [_conv1 computeWithCommandBuffer:commandBuffer
                               input:input
                          inputShape:inputShape
                              output:_buffer1
                         outputShape:inputShape];
    
    [_batchNorm1 computeWithCommandBuffer:commandBuffer
                                    input:_buffer1
                               inputShape:inputShape
                                   output:_buffer2
                              outputShape:inputShape];
    
    [_relu1 computeWithCommandBuffer:commandBuffer
                               input:_buffer2
                          inputShape:inputShape
                              output:_buffer1
                         outputShape:inputShape];
    
    [_conv2 computeWithCommandBuffer:commandBuffer
                               input:_buffer1
                          inputShape:inputShape
                              output:_buffer2
                         outputShape:inputShape];
    
    [_batchNorm2 computeWithCommandBuffer:commandBuffer
                                    input:_buffer2
                               inputShape:inputShape
                                   output:_buffer1
                              outputShape:inputShape];
    
    [_add computeWithCommandBuffer:commandBuffer
                            input1:_buffer1
                            input2:input
                            output:output
                             shape:inputShape];
}

@end









