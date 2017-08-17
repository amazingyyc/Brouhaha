/**
 * just for test
 *
 * Created by yanyuanchi on 2017/6/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

@import MetalPerformanceShaders;

#import <Metal/Metal.h>

#import "sys/mman.h"
#import "MPSLeNetViewController.h"
#import "BrouUtils.h"
#import "BrouConvertFloat.h"

@interface MPSLeNetViewController ()

@end

@implementation MPSLeNetViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    /**load the LeNet model to test the Brouhaha*/
    NSBundle *mainBundle = [NSBundle mainBundle];

    /**read the file*/
    void *w0 = [self readBinaryFile:[mainBundle pathForResource:@"w0" ofType:@""] length:20*1*5*5*4];
    void *b0 = [self readBinaryFile:[mainBundle pathForResource:@"b0" ofType:@""] length:20*4];
    void *w1 = [self readBinaryFile:[mainBundle pathForResource:@"w1" ofType:@""] length:50*20*5*5*4];
    void *b1 = [self readBinaryFile:[mainBundle pathForResource:@"b1" ofType:@""] length:50*4];
    void *w2 = [self readBinaryFile:[mainBundle pathForResource:@"w2" ofType:@""] length:500*800*4];
    void *b2 = [self readBinaryFile:[mainBundle pathForResource:@"b2" ofType:@""] length:500*4];
    void *w3 = [self readBinaryFile:[mainBundle pathForResource:@"w3" ofType:@""] length:10*500*4];
    void *b3 = [self readBinaryFile:[mainBundle pathForResource:@"b3" ofType:@""] length:10*4];

    /**read test data, the testX/testY include 100 test data*/
    void *testX  = [self readBinaryFile:[mainBundle pathForResource:@"testX100" ofType:@""] length:100*784*4];
    float *testY = [self readBinaryFile:[mainBundle pathForResource:@"testY100" ofType:@""] length:100*4];

    void *float16Input = malloc(28 * 28 * 2);
    convertFloat32ToFloat16((uint32_t*)testX, (uint16_t*)float16Input, 28 * 28);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    /**use the mps*/
    MPSCNNConvolutionDescriptor *con0Des = [MPSCNNConvolutionDescriptor
                                            cnnConvolutionDescriptorWithKernelWidth:5
                                            kernelHeight:5
                                            inputFeatureChannels:1
                                            outputFeatureChannels:20
                                            neuronFilter:nil];

    MPSCNNConvolution *con0 = [[MPSCNNConvolution alloc] initWithDevice:device
                                                  convolutionDescriptor:con0Des
                                                          kernelWeights:w0
                                                              biasTerms:NULL
                                                                  flags:MPSCNNConvolutionFlagsNone];

    MPSOffset offset;
    offset.x = 2;
    offset.y = 2;
    offset.z = 0;
    con0.offset = offset;


    MPSCNNPoolingMax *maxPooling0 = [[MPSCNNPoolingMax alloc] initWithDevice:device
                                                                 kernelWidth:2
                                                                kernelHeight:2
                                                             strideInPixelsX:2
                                                             strideInPixelsY:2];

    offset.x = 1;
    offset.y = 1;
    offset.z = 0;
    maxPooling0.offset = offset;

    MPSImageDescriptor *inputImageDes = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                                                       width:28
                                                                                      height:28
                                                                             featureChannels:1];

    MPSImage *inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputImageDes];

    MPSImageDescriptor *con0OutputDes = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                                                       width:24
                                                                                      height:24
                                                                             featureChannels:20];
    MPSImage *con0OutputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:con0OutputDes];

    MPSImageDescriptor *maxPooling0OuptuDes = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                                                             width:12
                                                                                            height:12
                                                                                   featureChannels:20];

    MPSImage *maxPooling0OuptuImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:maxPooling0OuptuDes];

    /**copy input to inputimage*/
    MTLRegion inputDataRegion = MTLRegionMake3D(0, 0, 0, 28, 28, 1);
    [inputImage.texture replaceRegion:inputDataRegion mipmapLevel:0 withBytes:float16Input bytesPerRow:28 * 2];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    [con0 encodeToCommandBuffer:commandBuffer sourceImage:inputImage destinationImage:con0OutputImage];
    [maxPooling0 encodeToCommandBuffer:commandBuffer sourceImage:con0OutputImage destinationImage:maxPooling0OuptuImage];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    /**copy data*/
    void *float16Output = malloc(12 * 12 * 20 * 2);
    float *realOutput = malloc(12 * 12 * 20 * 4);

    for (int i = 0; i < 5; ++i) {
        [maxPooling0OuptuImage.texture getBytes:float16Output + i*4*2*12*12
                                    bytesPerRow:12*2*4
                                  bytesPerImage:0
                                     fromRegion:MTLRegionMake3D(0, 0, 0, 12, 12, 1)
                                    mipmapLevel:0
                                          slice:i];
    }

    convertFloat16ToFloat32((uint16_t*)float16Output, (uint32_t*)realOutput, 12 * 12 * 20);

    /**
     * print the output just for test
     */
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            for (int k = 0; k < 20; ++k) {
                int xx = k / 4;
                int yy = k % 4;

                float tt = realOutput[12 * 12 * 4 * xx + (i * 12 + j) * 4 + yy];

                if (0 != tt) {
                    NSLog(@"yyc %f", tt);
                }
            }
        }
    }
}

/**
 * read binary file to memory
 */
- (void*)readBinaryFile:(NSString*)path length:(int)length {
    int file = open([path UTF8String], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);

    assert(file != -1);

    void * filePointer = mmap(nil, length, PROT_READ, MAP_FILE | MAP_SHARED, file, 0);

    close(file);

    return filePointer;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

@end













