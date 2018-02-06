#import "BrouLayer.h"

@implementation BrouLayer

- (instancetype)initWithName:(NSString *)name {
    self = [super init];
    
    if (self) {
        _name = name;
    }
    
    return self;
}

- (void)computeCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                       input:(id<BrouTensor>)input
                      output:(id<BrouTensor>)output {
    NSAssert(false, @"BrouLayer is base class, can't be inited directly");
}

@end
