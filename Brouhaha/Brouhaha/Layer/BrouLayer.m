#import "BrouLayer.h"

@implementation BrouLayer

- (instancetype)initWithName:(NSString *)name {
    self = [super init];
    
    if (self) {
        _name = name;
    }
    
    return self;
}

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library {
    id<MTLFunction> function = [library newFunctionWithName:_functionName];
    
    NSAssert(function, @"init %@ function:%@ error!", _name, _functionName);
    
    /**get the function*/
    NSError *error = nil;
    
    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    
    NSAssert(_computePipelineState, @"init %@ ComputePipelineState error:%@", _name, error);
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape {
    NSAssert(false, @"BrouLayer is base class, can't be inited directly");
}

- (void)setLayerName:(NSString*)name {
    _name = name;
}

- (NSString*)getLayerName {
    return _name;
}

@end
