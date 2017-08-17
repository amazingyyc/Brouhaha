/**
 * ViewController.m
 *
 * Created by yanyuanchi on 2017/5/10.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * just for test
 */

@import Metal;
@import MetalKit;
@import MetalPerformanceShaders;

#ifdef DEBUG
#define NSLog(FORMAT, ...) fprintf(stderr,"%s\n",[[NSString stringWithFormat:FORMAT, ##__VA_ARGS__] UTF8String]);
#else
#define NSLog(...)
#endif

#import "ViewController.h"
#import "BrouUtils.h"
#import "BrouMatrixMultipy.h"
#import "BrouMatrixMultiplyX4.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    int M = 1024;
    int K = 1024;
    int N = 1024;

    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    float *C = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < 1024 * 1024; ++i) {
        A[i] = B[i] = 1.4f;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    NSError *libraryError = NULL;
    NSString *libraryFile = [[NSBundle mainBundle] pathForResource:@"BrouhahaMetal" ofType:@"metallib"];
    id <MTLLibrary> library = [device newLibraryWithFile:libraryFile error:&libraryError];

    if (!library) {
        NSLog(@"Library error: %@", libraryError);
    }

    /**
     * MPS matrix multiply
     */
    MPSMatrixDescriptor *mpsMatrixDes = [MPSMatrixDescriptor new];

    size_t aRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:K dataType:MPSDataTypeFloat32];
    size_t bRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:N dataType:MPSDataTypeFloat32];
    size_t cRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:N dataType:MPSDataTypeFloat32];

    id<MTLBuffer> aBuffer = [device newBufferWithBytes:A
                                                length:sizeof(float) * M * K
                                               options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    id<MTLBuffer> bBuffer = [device newBufferWithBytes:B
                                                length:sizeof(float) * K * N
                                               options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    id<MTLBuffer> mpsCBuffer = [device newBufferWithLength:sizeof(float) * M * N
                                                   options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];

    mpsMatrixDes.rows = M;
    mpsMatrixDes.columns = K;
    mpsMatrixDes.dataType = MPSDataTypeFloat32;
    mpsMatrixDes.rowBytes = aRowBytes;

    MPSMatrix *aMatrix = [[MPSMatrix alloc] initWithBuffer:aBuffer descriptor:mpsMatrixDes];

    mpsMatrixDes.rows = K;
    mpsMatrixDes.columns = N;
    mpsMatrixDes.dataType = MPSDataTypeFloat32;
    mpsMatrixDes.rowBytes = bRowBytes;

    MPSMatrix *bMatrix = [[MPSMatrix alloc] initWithBuffer:bBuffer descriptor:mpsMatrixDes];

    mpsMatrixDes.rows = M;
    mpsMatrixDes.columns = N;
    mpsMatrixDes.dataType = MPSDataTypeFloat32;
    mpsMatrixDes.rowBytes = cRowBytes;

    MPSMatrix *cMatrix = [[MPSMatrix alloc] initWithBuffer:mpsCBuffer descriptor:mpsMatrixDes];

    MPSMatrixMultiplication *mpsMM = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                       transposeLeft:FALSE
                                                                      transposeRight:FALSE
                                                                          resultRows:M
                                                                       resultColumns:N
                                                                     interiorColumns:K
                                                                               alpha:1.0
                                                                                beta:0];

    id<MTLCommandBuffer> mpsBuffer = [queue commandBuffer];

    [mpsMM encodeToCommandBuffer:mpsBuffer leftMatrix:aMatrix rightMatrix:bMatrix resultMatrix:cMatrix];


    UInt64 t1, t2;
    t1 = [self getCurrentTimeNow];

    [mpsBuffer commit];
    [mpsBuffer waitUntilCompleted];

    t2 = [self getCurrentTimeNow];

    NSLog(@"yyc MPS time:%llu", (t2 - t1));

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    BrouMatrixMultiplyX4 *brouMatrixMultipy = [[BrouMatrixMultiplyX4 alloc] initWithDevice:device library:library M:M K:K N:N];
    [brouMatrixMultipy computeWithCommandBuffer:commandBuffer A:A B:B];

    t1 = [self getCurrentTimeNow];

    [brouMatrixMultipy copyInA:A];

    t2 = [self getCurrentTimeNow];

    NSLog(@"yyc BrouMatrixMultipy copy in A time:%llu", (t2 - t1));

    t1 = [self getCurrentTimeNow];

    [brouMatrixMultipy copyInB:B];

    t2 = [self getCurrentTimeNow];

    NSLog(@"yyc BrouMatrixMultipy copy in B time:%llu", (t2 - t1));

    t1 = [self getCurrentTimeNow];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    t2 = [self getCurrentTimeNow];

    NSLog(@"yyc BrouMatrixMultipy matrix multiply time:%llu", (t2 - t1));

    t1 = [self getCurrentTimeNow];

    [brouMatrixMultipy copyTOC:C];

    t2 = [self getCurrentTimeNow];

    NSLog(@"yyc BrouMatrixMultipy copy to C time:%llu", (t2 - t1));
}

- (UInt64)getCurrentTimeNow
{
    UInt64 recordTime = [[NSDate date] timeIntervalSince1970] * 1000;

    return recordTime;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}


@end











