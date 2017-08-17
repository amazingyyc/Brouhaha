/**
 * Created by yanyuanchi on 2017/7/23.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * a residual layer
 * include two convolution layer, two batch-norm layer, two relu layer
 */

#import "BrouConvolutionMMLayer.h"
#import "BrouBatchNormalizationLayer.h"
#import "BrouAddLayer.h"
#import "BrouReLuLayer.h"
#import "BrouResidualLayer.h"

@interface BrouResidualLayer() {
    BrouConvolutionMMLayer *_conv1;
    BrouBatchNormalizationLayer *_batchNorm1;
    BrouReLuLayer *_relu1;

    BrouConvolutionMMLayer *_conv2;
    BrouBatchNormalizationLayer *_batchNorm2;
    
    BrouAddLayer *_add;
    
    id<MTLBuffer> _buffer1;
    id<MTLBuffer> _buffer2;
    
    int _height;
    int _width;
}
@end

@implementation BrouResidualLayer

/**
 * this residual layer ref:https://github.com/lengstrom/fast-style-transfer#video-stylization
 * the input diemsion is (height, width, 128)
 * the output dimension is (height, width, 128)
 * the kernel dimension is (128, 3, 3, 128)
 * the stride is (1, 1)
 * the pad is (1, 1)
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                float32Weight1:(void*)weight1
                float32Weight2:(void*)weight2
                 float32Alpha1:(void*)alpha1
                  float32Beta1:(void*)beta1
                 float32Alpha2:(void*)alpha2
                  float32Beta2:(void*)beta2
                        height:(int)height
                         width:(int)width {
    self = [super initWithLabel:@"BrouResidualLayer"];
    
    if (!self) {
        return self;
    }
    
    _conv1 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                           library:library
                                                            kernel:weight1
                                                              bias:nil
                                                       inputHeight:height
                                                        inputWidth:width
                                                     intputChannel:128
                                                      outputHeight:height
                                                       outputWidth:width
                                                     outputChannel:128
                                                      kernelHeight:3
                                                       kernelWidth:3
                                                           padLeft:1
                                                            padTop:1
                                                           strideX:1
                                                           strideY:1];
    
    _batchNorm1 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                     library:library
                                                                       alpha:alpha1
                                                                        beta:beta1
                                                                     epsilon:0.001
                                                                      height:height
                                                                       width:width
                                                                     channel:128];
    
    _relu1 = [[BrouReLuLayer alloc] initReLuWithDevice:device library:library height:height width:width channel:128];
    
    _conv2 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                           library:library
                                                            kernel:weight2
                                                              bias:nil
                                                       inputHeight:height
                                                        inputWidth:width
                                                     intputChannel:128
                                                      outputHeight:height
                                                       outputWidth:width
                                                     outputChannel:128
                                                      kernelHeight:3
                                                       kernelWidth:3
                                                           padLeft:1
                                                            padTop:1
                                                           strideX:1
                                                           strideY:1];
    
    
    _batchNorm2 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                     library:library
                                                                       alpha:alpha2
                                                                        beta:beta2
                                                                     epsilon:0.001
                                                                      height:height
                                                                       width:width
                                                                     channel:128];

    NSArray<NSNumber*> *dim1 = @[@(height), @(width), @(128)];
    NSArray<NSNumber*> *dim2 = @[@(height), @(width), @(128)];
    
    _add = [[BrouAddLayer alloc] initWithDevice:device library:library in1Dim:dim1 in2Dim:dim2];
    
    _buffer1 = [device newBufferWithLength:height * width * 128 * 2
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _buffer2 = [device newBufferWithLength:height * width * 128 * 2
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _height = height;
    _width  = width;
    
    return self;
}


/**
 * return the output bytes
 */
- (int)getOutputBytes {
    return _height * _width * 128 * 2;
}

/**
 * get the bytes of input
 */
- (int)getInputBytes {
    return _height * _width * 128 * 2;
}

/**
 compute the operator
 
 @param commandBuffer the command buffer of mrtal
 @param input input buffer
 @param output output buffer
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    [_conv1 computeWithCommandBuffer:commandBuffer input:input output:_buffer1];
    [_batchNorm1 computeWithCommandBuffer:commandBuffer input:_buffer1 output:_buffer2];
    [_relu1 computeWithCommandBuffer:commandBuffer input:_buffer2 output:_buffer1];
    [_conv2 computeWithCommandBuffer:commandBuffer input:_buffer1 output:_buffer2];
    [_batchNorm2 computeWithCommandBuffer:commandBuffer input:_buffer2 output:_buffer1];
    [_add computeWithCommandBuffer:commandBuffer input1:_buffer1 input2:input output:output];
}


@end









