#import "Brouhaha.h"

@interface BrouResidualLayer_float : BrouLayer

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
