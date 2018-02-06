#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/**
 this store a shared buffer of Metal
 */
@interface BrouShareBuffer : NSObject

+ (instancetype)defaultWithDevice:(id<MTLDevice>)device;
+ (instancetype)initWithLifeTime:(NSUInteger)time device:(id<MTLDevice>)device;

- (NSNumber*)bindWithBytesCounts:(NSUInteger)bytesCount;
- (id<MTLBuffer>)getBindBufferById:(NSNumber*)bindId;

@end
