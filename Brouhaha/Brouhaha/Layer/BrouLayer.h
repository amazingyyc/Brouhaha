#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "BrouStruct.h"
#import "BrouTensor.h"

@interface BrouLayer : NSObject

@property(nonatomic, strong) NSString *name;

- (instancetype)initWithName:(NSString *)name;

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output;
@end
