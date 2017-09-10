#import "Brouhaha.h"

@interface BrouResidualLayer_half : BrouLayer {
    BrouConvolutionMMLayer_half *_conv1;
    BrouBatchNormalizationLayer_half *_batchNorm1;
    BrouReLuLayer_half *_relu1;
    
    BrouConvolutionMMLayer_half *_conv2;
    BrouBatchNormalizationLayer_half *_batchNorm2;
    
    BrouAddLayer_half *_add;
    
    int _channel;
    int _channelX4;
    
    id<MTLBuffer> _buffer1;
    id<MTLBuffer> _buffer2;
}

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
                  floatWeight1:(void*)floatWeight1
                  floatWeight2:(void*)floatWeight2
                   floatAlpha1:(void*)floatAlpha1
                    floatBeta1:(void*)floatBeta1
                   floatAlpha2:(void*)floatAlpha2
                    floatBeta2:(void*)floatBeta2
                       channel:(int)channel;

@end
