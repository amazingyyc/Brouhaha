/**
 * BrouLayer.h
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the protocol of all layers
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/**
 * store the shpe of data
 * (dim0,dim1, dim2) represent a 3d dimension
 */
typedef struct _TensorShape {
    int32_t dim0;
    int32_t dim1;
    int32_t dim2;
} TensorShape;

/**
 * the data's dimension type
 */
typedef NS_ENUM(NSInteger, DimensionType) {
    Dimension1D = 1,
    Dimension2D = 2,
    Dimension3D = 3,
    Dimension4D = 4
};

@interface BrouLayer : NSObject {
    NSString *_label;
    NSString *_functionName;

    id<MTLComputePipelineState> _computePipelineState;
}

- (instancetype)initWithLabel:(NSString*)label;

/**
 compute the operator

 @param commandBuffer the command buffer of mrtal
 @param input input buffer
 @param output output buffer
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output;

/**
 * return the output bytes
 */
- (int)getOutputBytes;

/**
 * get the bytes of input
 */
- (int)getInputBytes;

@end
