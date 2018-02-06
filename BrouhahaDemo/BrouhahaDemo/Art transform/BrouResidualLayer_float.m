#import "BrouResidualLayer_float.h"

@interface BrouResidualLayer_float() {
    BrouConvolutionMMLayer_float *_conv1;
    BrouBatchNormalizationLayer_float *_batchNorm1;
    BrouReLuLayer_float *_relu1;
    
    BrouConvolutionMMLayer_float *_conv2;
    BrouBatchNormalizationLayer_float *_batchNorm2;
    
    BrouAddLayer_float *_add;
    
    int _channel;
    int _channelX4;
    
    id<BrouTensor> _buffer1;
    id<BrouTensor> _buffer2;
}

@end

@implementation BrouResidualLayer_float

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  floatWeight1:(void*)floatWeight1
                  floatWeight2:(void*)floatWeight2
                   floatAlpha1:(void*)floatAlpha1
                    floatBeta1:(void*)floatBeta1
                   floatAlpha2:(void*)floatAlpha2
                    floatBeta2:(void*)floatBeta2
                       channel:(int)channel {
    self = [super initWithName:@"BrouResidualLayer_float"];
    
    if (!self) {
        return self;
    }
    
    _channel   = channel;
    _channelX4 = (_channel + 3) / 4 * 4;
    
    _conv1 = [[BrouConvolutionMMLayer_float alloc] initWithDevice:device
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
    
    _batchNorm1 = [[BrouBatchNormalizationLayer_float alloc] initWithDevice:device
                                                                    library:library
                                                                    epsilon:0.001
                                                                 floatAlpha:floatAlpha1
                                                                  floatBeta:floatBeta1
                                                                    channel:_channel];
    
    _relu1 = [[BrouReLuLayer_float alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    _conv2 = [[BrouConvolutionMMLayer_float alloc] initWithDevice:device
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
    
    _batchNorm2 = [[BrouBatchNormalizationLayer_float alloc] initWithDevice:device
                                                                    library:library
                                                                    epsilon:0.001
                                                                 floatAlpha:floatAlpha2
                                                                  floatBeta:floatBeta2
                                                                    channel:_channel];
    
    _add = [[BrouAddLayer_float alloc] initWithDevice:device library:library dimensionType:Dimension3D];

    return self;
}

- (void)checkParamsWithInput:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    NSAssert(input.dim0 == output.dim0 && input.dim0 > 0 &&
             input.dim1 == output.dim1 && input.dim1 > 0 &&
             input.dim2 == output.dim2 && input.dim2 > 0,
             @"the shape is error!");
    
    NSAssert(input.innermostDimX4 == _channelX4, @"the shape is error!");
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    [self checkParamsWithInput:input output:output];
    
    NSUInteger length = sizeof(float) * input.dim0 * input.dim1 * input.innermostDimX4;
    
    if (!_buffer1 || _buffer1.tensorBuffer.length < length) {
        _buffer1 = [BrouUniqueTensor_float initWithHeight:input.dim0
                                                    width:input.dim1
                                                  channel:input.innermostDimX4
                                                   device:commandBuffer.device];
    }
    
    if (!_buffer2 || _buffer2.tensorBuffer.length < length) {
        _buffer2 = [BrouUniqueTensor_float initWithHeight:input.dim0
                                                    width:input.dim1
                                                  channel:input.innermostDimX4
                                                   device:commandBuffer.device];
    }
    
    [_conv1      computeCommandBuffer:commandBuffer input:input    output:_buffer1];
    [_batchNorm1 computeCommandBuffer:commandBuffer input:_buffer1 output:_buffer2];
    [_relu1      computeCommandBuffer:commandBuffer input:_buffer2 output:_buffer1];
    [_conv2      computeCommandBuffer:commandBuffer input:_buffer1 output:_buffer2];
    [_batchNorm2 computeCommandBuffer:commandBuffer input:_buffer2 output:_buffer1];
    [_add computeCommandBuffer:commandBuffer input1:_buffer1 input2:input output:output];
}

@end









