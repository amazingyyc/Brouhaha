#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface BrouTemporaryBuffer : NSObject {
}

@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic, assign) int length;

- (instancetype)init;

- (void)configWithFloatLength:(int)length;
- (void)configWithFloatHeight:(int)height width:(int)width;
- (void)configWithFloatHeight:(int)height width:(int)width channel:(int)channel;

- (void)configWithHalfLength:(int)length;
- (void)configWithHalfHeight:(int)height width:(int)width;
- (void)configWithHalfHeight:(int)height width:(int)width channel:(int)channel;

/**
 * for the ConvlutionMM, TransposedConvolutionMM, DilatedConvolutionMM
 * the output (height, width, channel) must be subject to: 0 == (height * width) % 4 and 0 == channel % 4
 */
- (void)configConvolutionMMWithFloatHeight:(int)height width:(int)width channel:(int)channel;
- (void)configConvolutionMMWithHalfHeight:(int)height width:(int)width channel:(int)channel;


- (void)configWithDevice:(id<MTLDevice>)device;

@end
