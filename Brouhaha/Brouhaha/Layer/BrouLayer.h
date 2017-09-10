#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "BrouStruct.h"

@interface BrouLayer : NSObject {
    /**the name of the layer*/
    NSString *_name;
    
    /**the name of Metal fucntion*/
    NSString *_functionName;
    
    /**the Metal computePipelineState*/
    id<MTLComputePipelineState> _computePipelineState;
}

- (instancetype)initWithName:(NSString*)name;

- (void)configComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                      library:(id<MTLLibrary>)library;

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                      inputShape:(TensorShape)inputShape
                          output:(id<MTLBuffer>)output
                     outputShape:(TensorShape)outputShape;

- (void)setLayerName:(NSString*)name;
- (NSString*)getLayerName;

@end
