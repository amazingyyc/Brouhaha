/**
 * BrouLinearLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the linear layer
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouLinearLayer.h"

/**
 * store the a and b
 */
typedef struct _AB {
    uint16_t a;
    uint16_t b;
} AB;

@interface BrouLinearLayer() {
    /**
     * the dimension must be 1, 2, 3
     */
    DimensionType _dimensionType;
    
    /**store the shape*/
    id<MTLBuffer> _shape;
    id<MTLBuffer> _ab;
    
    float _float32A;
    float _float32B;
    
    uint16_t _float16A;
    uint16_t _float16B;
    
    int _len;
}

@end

@implementation BrouLinearLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)lirary
                        height:(int)height
                         width:(int)width
                       channel:(int)channel
                             a:(float)a
                             b:(float)b {
    self = [super initWithLabel:@"BrouLinearLayer"];
    
    if (!self) {
        return self;
    }
    
    _float32A = a;
    _float32B = b;
    
    _float16A = convertFloat32ToFloat16OneNumber((uint32_t*)(&_float32A));
    _float16B = convertFloat32ToFloat16OneNumber((uint32_t*)(&_float32B));
    
    int channelX4 = (channel + 3) / 4 * 4;
    
    _len = height * width * channelX4;
    
    _dimensionType = Dimension3D;
    _functionName  = @"brouLinear3D";
    
    _shape = [device newBufferWithLength:sizeof(TensorShape)
                                 options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _ab = [device newBufferWithLength:sizeof(AB)
                              options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    shapeRef->dim0 = height;
    shapeRef->dim1 = width;
    shapeRef->dim2 = channelX4;
    
    AB *abRef = (AB*)_ab.contents;
    abRef->a = _float16A;
    abRef->b = _float16B;
    
    [self buildComputePipelinesStateWithDevice:device library:lirary];
    
    return self;
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {

    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:_functionName];

    if (!function) {
        NSLog(@"init  function error");

        return;
    }

    _computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];

    if (!_computePipelineState) {
        NSLog(@"init MTLComputePipelineState error");
    }
}

/**
 compute the operator
 @param commandBuffer the command buffer of mrtal
 @param input input buffer
 @param output output buffer

 the input's dimension should be (inputHeight, intputWidth, channelX4)
 the output's dimension should be (outputHeight, outputWidth, channelX4)
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];

    [encoder setBuffer:input  offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:_ab    offset:0 atIndex:2];
    [encoder setBuffer:_shape offset:0 atIndex:3];

    NSUInteger exeWidth = _computePipelineState.threadExecutionWidth;
    MTLSize threadsPerThreadgroup = MTLSizeMake(0, 0, 0);
    MTLSize threadgroupsPerGrid   = MTLSizeMake(0, 0, 0);

    TensorShape *shapeRef = (TensorShape*)_shape.contents;
    
    if (_dimensionType == Dimension1D) {
        threadsPerThreadgroup = MTLSizeMake(exeWidth, 1, 1);
        threadgroupsPerGrid   = MTLSizeMake((shapeRef->dim0 + exeWidth * 4 - 1) / (exeWidth * 4),
                                            1,
                                            1);
    } else if (_dimensionType == Dimension2D) {
        threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
        threadgroupsPerGrid   = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                                            (shapeRef->dim0 + 15) / 16,
                                            1);
    } else if (_dimensionType == Dimension3D) {
        threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
        threadgroupsPerGrid   = MTLSizeMake((shapeRef->dim1 + 31) / 32,
                                            (shapeRef->dim0 + 15) / 16,
                                            (shapeRef->dim2 / 4));
    } else {
        /**todo support all dimension data*/
        NSAssert(false, @"The data dimension is error");
    }
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

/**
 * return the bytes of output
 */
- (int)getOutputBytes {
    return _len * 2;
}

- (int)getInputBytes {
    return _len * 2;
}

@end










