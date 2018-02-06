#import <Foundation/Foundation.h>

/**
 this the delegate of BrouTensor and BrouTemporaryTensor
 */

@protocol BrouTensor <NSObject>

- (NSUInteger)dimension;

- (int)height;
- (int)width;
- (int)channel;

- (int)dim0;
- (int)dim1;
- (int)dim2;

- (int)dim:(int)dim;

- (int)innermostDim;
- (int)innermostDimX4;

- (id<MTLBuffer>)tensorBuffer;

@end
