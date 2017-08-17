/**
 * for compatible the other layers the addlayer have some constrain
 * for 1d 2d 3d tensor the diemnsion is (dim0) (dim0, dim1) (dim0, dim1, dim2)
 *
 * the last dim of in tensor must be 1 or time by 4, and can not be 1 at the same time
 * for in1(31, 1) in2(1, 1) can be changed to in1(31) in2(1)
 *
 * like 1d tensor the dimension of in1, in2, out is (in1-dim0) (in2-dim0) (out-dim0)
 * the in1-dim0 and in2-dim0 must be 1 or timed by 4 and in1-dim0 and in2-dim0 can not be 1 at same time
 */

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouAddLayer.h"

@interface BrouAddLayer() {
    /**the dimension must be 1, 2, 3*/
    DimensionType _dimensionType;
    
    NSMutableArray<NSNumber*> *_in1Dim;
    NSMutableArray<NSNumber*> *_in2Dim;
    
    /**for compatible other layer the last dimension of in and out must be timed by 4*/
    NSMutableArray<NSNumber*> *_in1DimX4;
    NSMutableArray<NSNumber*> *_in2DimX4;
    
    NSMutableArray<NSNumber*> *_outDim;
    
    id<MTLBuffer> _in1ShapeBuffer;
    id<MTLBuffer> _in2ShapeBuffer;
    id<MTLBuffer> _outShapeBuffer;
    
    /**
     * if this addlayer is used to add a bias to a 1d, 2d, 3d input
     * than when inited it should take a bias in
     */
    id<MTLBuffer> _bias;
}

@end;

@implementation BrouAddLayer

/**
 * this init function can be used to add two input buffer
 * like in resnet, a input(dim0, dim1, dim2) add a another input (dim0, dim1, dim2)
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                        in1Dim:(NSArray<NSNumber*>*) in1Dim
                        in2Dim:(NSArray<NSNumber*>*) in2Dim {
    self = [super initWithLabel:@"BrouAddLayer"];
    
    if (!self) {
        return self;
    }
    
    [self configDimWithDevice:device in1Dim:in1Dim in2Dim:in2Dim];
    [self buildComputePipelinesStateWithDevice:device library:library];
    
    return self;
}

/**
 * this init function can be used to add a bias to a 3d-image or a matrix or a 1d-array
 *
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                   float32Bias:(void*)bias
                       biasDim:(NSArray<NSNumber*>*) biasDim
                         inDim:(NSArray<NSNumber*>*) inDim {
    self = [super initWithLabel:@"BrouAddLayer"];
    
    if (!self) {
        return self;
    }
    
    [self configDimWithDevice:device in1Dim:biasDim in2Dim:inDim];
    [self buildComputePipelinesStateWithDevice:device library:library];
    
    int len = 1;
    int lenx4 = 1;
    
    for (NSNumber *n in _in1Dim) {
        len *= n.intValue;
    }
    
    for (NSNumber *n in _in1DimX4) {
        lenx4 *= n.intValue;
    }
    
    /**store the bias*/
    _bias = [device newBufferWithLength:lenx4 * 2
                                options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    uint16_t *float16Bias = malloc(len * 2);
    
    convertFloat32ToFloat16((uint32_t*)bias, float16Bias, len);
    
    [self copyBiasWithFloat16Bias:float16Bias toBuffer:_bias.contents];
    
    free(float16Bias);
    
    return self;
}

- (void)copyBiasWithFloat16Bias:(void*)float16Bias toBuffer:(void*)buffer {
    int rowByte   = _in1Dim.lastObject.intValue * 2;
    int rowByteX4 = _in1DimX4.lastObject.intValue * 2;
    int zeroByte  = rowByteX4 - rowByte;
    
    int iter = 1;
    
    for (int i = 0; i < _in1Dim.count - 1; ++i) {
        iter *= _in1Dim[i].intValue;
    }
    
    for (int i = 0; i < iter; ++i) {
        memcpy(buffer + i * rowByteX4, float16Bias + i * rowByte, rowByte);
        memset(buffer + i * rowByteX4 + rowByte, 0, zeroByte);
    }
}

- (void)configDimWithDevice:(id<MTLDevice>)device
                     in1Dim:(NSArray<NSNumber*>*) in1Dim
                     in2Dim:(NSArray<NSNumber*>*) in2Dim {
    NSAssert(nil != in1Dim && 0 != in1Dim.count && 3 >= in1Dim.count, @"input1Dim count must be > 0 and <= 3");
    NSAssert(nil != in2Dim && 0 != in2Dim.count && 3 >= in2Dim.count, @"input2Dim count must be > 0 and <= 3");
    
    int dimSize = (int)MAX(in1Dim.count, in2Dim.count);
    
    _in1Dim = [in1Dim mutableCopy];
    _in2Dim = [in2Dim mutableCopy];
    
    _outDim = [[NSMutableArray<NSNumber*> alloc] init];
    
    /**add 1 to _in1Dim and in2Dim*/
    for (int i = 0; i < dimSize - _in1Dim.count; ++i) {
        [_in1Dim insertObject:[[NSNumber alloc] initWithInt:1] atIndex:0];
    }
    
    for (int i = 0; i < dimSize - in2Dim.count; ++i) {
        [_in2Dim insertObject:[[NSNumber alloc] initWithInt:1] atIndex:0];
    }
    
    _in1DimX4 = [_in1Dim mutableCopy];
    _in2DimX4 = [_in2Dim mutableCopy];
    
    int in1LastX4 = (_in1DimX4.lastObject.intValue + 3) / 4 * 4;
    int in2LastX4 = (_in2DimX4.lastObject.intValue + 3) / 4 * 4;
    
    _in1DimX4[dimSize - 1] = [[NSNumber alloc] initWithInt:in1LastX4];
    _in2DimX4[dimSize - 1] = [[NSNumber alloc] initWithInt:in2LastX4];
    
    for (int i = dimSize - 1; i >= 0; --i) {
        if (1 != _in1DimX4[i].integerValue && 1 != _in2DimX4[i].integerValue && _in1DimX4[i].integerValue != _in2DimX4[i].integerValue) {
            NSAssert(NO, @"the input dimension is error");
        }
        
        NSNumber *number = [[NSNumber alloc] initWithLong:MAX(_in1DimX4[i].integerValue, _in2DimX4[i].integerValue)];
        
        [_outDim insertObject:number atIndex:0];
    }
    
    /**init dim buffer*/
    _in1ShapeBuffer = [device newBufferWithLength:sizeof(TensorShape)
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _in2ShapeBuffer = [device newBufferWithLength:sizeof(TensorShape)
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    _outShapeBuffer = [device newBufferWithLength:sizeof(TensorShape)
                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    
    TensorShape *in1ShapeBufferRef = (TensorShape *)_in1ShapeBuffer.contents;
    TensorShape *in2ShapeBufferRef = (TensorShape *)_in2ShapeBuffer.contents;
    TensorShape *outShapeBufferRef = (TensorShape *)_outShapeBuffer.contents;
    
    if (1 == dimSize) {
        _dimensionType = Dimension1D;
        _functionName  = @"brouAdd1D";
        
        in1ShapeBufferRef->dim0 = _in1DimX4[0].intValue;
        in2ShapeBufferRef->dim0 = _in2DimX4[0].intValue;
        outShapeBufferRef->dim0 = _outDim[0].intValue;
    } else if (2 == dimSize) {
        _dimensionType = Dimension2D;
        _functionName  = @"brouAdd2D";
        
        in1ShapeBufferRef->dim0 = _in1DimX4[0].intValue;
        in2ShapeBufferRef->dim0 = _in2DimX4[0].intValue;
        outShapeBufferRef->dim0 = _outDim[0].intValue;
        
        in1ShapeBufferRef->dim1 = _in1DimX4[1].intValue;
        in2ShapeBufferRef->dim1 = _in2DimX4[1].intValue;
        outShapeBufferRef->dim1 = _outDim[1].intValue;
    } else if (3 == dimSize) {
        _dimensionType = Dimension3D;
        _functionName  = @"brouAdd3D";
        
        in1ShapeBufferRef->dim0 = _in1DimX4[0].intValue;
        in2ShapeBufferRef->dim0 = _in2DimX4[0].intValue;
        outShapeBufferRef->dim0 = _outDim[0].intValue;
        
        in1ShapeBufferRef->dim1 = _in1DimX4[1].intValue;
        in2ShapeBufferRef->dim1 = _in2DimX4[1].intValue;
        outShapeBufferRef->dim1 = _outDim[1].intValue;
        
        in1ShapeBufferRef->dim2 = _in1DimX4[2].intValue;
        in2ShapeBufferRef->dim2 = _in2DimX4[2].intValue;
        outShapeBufferRef->dim2 = _outDim[2].intValue;
    }
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

- (int)getOutputBytes {
    int len = 1;
    
    for (NSNumber *n in _outDim) {
        len *= n.intValue;
    }
    
    return len;
}

- (int)getInputBytes {
    int len = 1;
    
    for (NSNumber *n in _in2DimX4) {
        len *= n.intValue;
    }
    
    return len;
}

/**
 compute the operator
 @param commandBuffer the command buffer of mrtal
 @param input input buffer
 @param output output buffer
 */
- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           input:(id<MTLBuffer>)input
                          output:(id<MTLBuffer>)output {
    
    [self computeWithCommandBuffer:commandBuffer input1:_bias input2:input output:output];
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                          input1:(id<MTLBuffer>)input1
                          input2:(id<MTLBuffer>)input2
                          output:(id<MTLBuffer>)output {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    
    [encoder setBuffer:input1  offset:0 atIndex:0];
    [encoder setBuffer:input2  offset:0 atIndex:1];
    [encoder setBuffer:output  offset:0 atIndex:2];
    [encoder setBuffer:_in1ShapeBuffer  offset:0 atIndex:3];
    [encoder setBuffer:_in2ShapeBuffer  offset:0 atIndex:4];
    [encoder setBuffer:_outShapeBuffer  offset:0 atIndex:5];

    NSUInteger exeWidth = _computePipelineState.threadExecutionWidth;
    MTLSize threadsPerThreadgroup;
    MTLSize threadgroupsPerGrid;
    
    if (_dimensionType == Dimension1D) {
        threadsPerThreadgroup = MTLSizeMake(exeWidth, 1, 1);
        threadgroupsPerGrid   = MTLSizeMake((_outDim[0].intValue + exeWidth * 4 - 1) / (exeWidth * 4),
                                            1, 1);
    } else if (_dimensionType == Dimension2D) {
        threadsPerThreadgroup = MTLSizeMake(1, exeWidth, 1);
        threadgroupsPerGrid   = MTLSizeMake(1,
                                            (_outDim[0].intValue + exeWidth * 4 - 1) / (exeWidth * 4),
                                            1);
    } else {
        threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
        threadgroupsPerGrid   = MTLSizeMake((_outDim[1].intValue + 31) / 32,
                                            (_outDim[0].intValue + 15) / 16,
                                            (_outDim[2].intValue / 4));
    }
    
    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

@end















