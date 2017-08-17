/**
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * just for test
 */

#import "BrouMatrixMultiplyX4.h"

/**
 * calculate the matrix A times matrix B
 * C = A * B
 * the A is (_M, _K)
 * the B is (_K, _N)
 * the C is (_M, _N)
 */
@interface BrouMatrixMultiplyX4() {
    /**normalize the _M to the times of 4*/
    int _MX4;

    /**normalize the _N to the times of 4*/
    int _NX4;

    /**store the matrix of A and B*/
    id<MTLBuffer> _aBuffer;
    id<MTLBuffer> _bBuffer;
    id<MTLBuffer> _cBuffer;
}

@property int M;
@property int K;
@property int N;

@property(nonatomic, copy) NSString *functionName;

@property(nonatomic, strong) id<MTLComputePipelineState> computePipelineState;

@end

@implementation BrouMatrixMultiplyX4

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                             M:(int)M
                             K:(int)K
                             N:(int)N {
    self = [super init];

    if (!self) {
        return self;
    }

    _M = M;
    _K = K;
    _N = N;

    _MX4 = (_M + 3) / 4 * 4;
    _NX4 = (_N + 3) / 4 * 4;

    _aBuffer = [device newBufferWithLength:_K * _MX4 * sizeof(float)
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    _bBuffer = [device newBufferWithLength:_K * _NX4 * sizeof(float)
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    _cBuffer = [device newBufferWithLength:_MX4 * _NX4 * sizeof(float)
                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    _functionName = @"brouMatrixMultiplyFloat32SpliteBlockX4";

    [self buildComputePipelinesStateWithDevice:device library:library];

    return self;
}

/**
 * build the computepipelinesstate
 */
- (void)buildComputePipelinesStateWithDevice:(id<MTLDevice>)device
                                     library:(id<MTLLibrary>)library {

    int aRowByte = _MX4 * sizeof(float);
    int bRowByte = _NX4 * sizeof(float);

    /**set the function constant*/
    MTLFunctionConstantValues *constantValues = [MTLFunctionConstantValues new];
    [constantValues setConstantValue:&_M type:MTLDataTypeInt atIndex:0];
    [constantValues setConstantValue:&_K type:MTLDataTypeInt atIndex:1];
    [constantValues setConstantValue:&_N type:MTLDataTypeInt atIndex:2];
    [constantValues setConstantValue:&aRowByte type:MTLDataTypeInt atIndex:3];
    [constantValues setConstantValue:&bRowByte type:MTLDataTypeInt atIndex:4];

    /**get the function*/
    NSError *error = nil;

    id<MTLFunction> function = [library newFunctionWithName:_functionName constantValues:constantValues error:&error];

    if (!function) {
        NSLog(@"init  function error");

        return;
    }

    self.computePipelineState = [device newComputePipelineStateWithFunction:function error:&error];

    if (!self.computePipelineState) {
        NSLog(@"init MTLComputePipelineState error");
    }
}

- (void)copyInA:(float*)A {
    for (int i = 0; i < _K; ++i) {
        float *aBufferData = _aBuffer.contents + _MX4 * sizeof(float) * i;
        float *aData       = A + i;

        for (int j = 0; j < _M; ++j) {
            aBufferData[j] = aData[0];

            aData += _K;
        }
    }
}

- (void)copyInB:(float*)B {
    int length = _N * sizeof(float);

    for (int i = 0; i < _K; ++i) {
        memcpy(_bBuffer.contents + _NX4 * sizeof(float) * i, B + _N * i, length);
    }
}

/**
 * copy data form cBuffer to C
 */
- (void)copyTOC:(float*)C {
    int length = _N * sizeof(float);

    for (int i = 0; i < _M; ++i) {
        memcpy(C + _N * i, _cBuffer.contents + _NX4 * sizeof(float) * i, length);
    }
}

- (void)computeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                               A:(float*)A
                               B:(float*)B {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_computePipelineState];
    [encoder setBuffer:_aBuffer offset:0 atIndex:0];
    [encoder setBuffer:_bBuffer offset:0 atIndex:1];
    [encoder setBuffer:_cBuffer offset:0 atIndex:2];

    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 4, 1);
    MTLSize threadgroupsPerGrid   = MTLSizeMake((_NX4 + 31) / (4 * 8), (_MX4 + 15) / (4 * 4), 1);

    [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

    [encoder endEncoding];
}


@end
