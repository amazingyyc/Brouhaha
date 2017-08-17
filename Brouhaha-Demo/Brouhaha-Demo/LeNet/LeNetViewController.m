/**
 * Created by yanyuanchi on 2017/6/18.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * use the LeNet to recgonize the digit number
 */

#import "sys/mman.h"

#import "LeNetViewController.h"

#import "BrouUtils.h"
#import "BrouConvertFloat.h"
#import "BrouConvolutionLayer.h"
#import "BrouConvolutionMMLayer.h"
#import "BrouMaxPoolingLayer.h"
#import "BrouFullConnectLayer.h"
#import "BrouTanHLayer.h"
#import "BrouAddLayer.h"
#import "BrouNet.h"
#import "BrouAddLayer.h"
#import "PaintView.h"

@interface LeNetViewController ()

@property(nonatomic, strong) PaintView *paintView;

@property(nonatomic, strong) UILabel *label;

@property(nonatomic, strong) UILabel *result;

@property(nonatomic, strong) UIButton *closeButton;
@property(nonatomic, strong) UIButton *clearButton;
@property(nonatomic, strong) UIButton *recognizeButton;

@property(nonatomic, strong) BrouNet *net;

@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;

@end

@implementation LeNetViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    CGRect bounds = self.view.bounds;
    
    _label = [[UILabel alloc] init];
    _label.font = [UIFont systemFontOfSize:15];
    _label.text = @"draw digit number (0~9) below and click \"Recognize\" button";
    _label.frame = CGRectMake(0, 20, bounds.size.width, 20);
    _label.numberOfLines = 0;
    _label.textColor = [UIColor whiteColor];
    [_label sizeToFit];
    
    _result = [[UILabel alloc] init];
    _result.font = [UIFont systemFontOfSize:15];
    _result.frame = CGRectMake(0, 60, bounds.size.width, 20);
    _result.numberOfLines = 0;
    _result.textColor = [UIColor redColor];
    
    CGFloat delta = bounds.size.width / 3;
    CGFloat y = (bounds.size.height - bounds.size.width) / 2 + bounds.size.width;
    
    _closeButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [_closeButton setTitle:@"Close" forState:UIControlStateNormal];
    _closeButton.frame = CGRectMake(0, y, delta, 50);
    
    _recognizeButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [_recognizeButton setTitle:@"Recognize" forState:UIControlStateNormal];
    _recognizeButton.frame = CGRectMake(delta, y, delta, 50);
    
    _clearButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [_clearButton setTitle:@"Clear" forState:UIControlStateNormal];
    _clearButton.frame = CGRectMake(2 * delta, y, delta, 50);
    
    _paintView = [[PaintView alloc] initWithFrame:CGRectMake(0, (bounds.size.height - bounds.size.width) / 2,
                                                             bounds.size.width, bounds.size.width)];
    [self.view addSubview:_label];
    [self.view addSubview:_result];
    [self.view addSubview:_closeButton];
    [self.view addSubview:_recognizeButton];
    [self.view addSubview:_clearButton];
    [self.view addSubview:_paintView];
    [self.view setBackgroundColor:[UIColor darkGrayColor]];
    
    [_closeButton     addTarget:self action:@selector(click:) forControlEvents:UIControlEventTouchUpInside];
    [_recognizeButton addTarget:self action:@selector(click:) forControlEvents:UIControlEventTouchUpInside];
    [_clearButton     addTarget:self action:@selector(click:) forControlEvents:UIControlEventTouchUpInside];
    
    /**init LeNet*/
    [self buildLeNet];
}

/**
 * recognize digit number from PaintView
 */
- (void)recognizeDigitNumber {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    void *rawData = malloc(28 * 28);
    
    CGContextRef context = CGBitmapContextCreate(rawData,
                                                 28,
                                                 28,
                                                 8,
                                                 28,
                                                 colorSpace,
                                                 kCGImageAlphaNone);

    CGContextTranslateCTM(context, 0, 28);
    CGContextScaleCTM(context, 28.0 / _paintView.frame.size.width, -28.0 / _paintView.frame.size.height);
    [_paintView.layer drawInContext:context];
    
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);

    uint16_t *float16Data   = malloc(28 * 28 * 2 * 4);
    uint16_t *float16Output = malloc(10 * 2);
    float *realOutput       = malloc(10 * 4);
    
    for (int i = 0; i < 28 * 28; ++i) {
        float float32 = ((uint8_t*)rawData)[i];
        float32 = 1.0 - float32 / 255.0;
        
        float16Data[i * 4] = convertFloat32ToFloat16OneNumber((uint32_t*)(&float32));
    }
    
    UInt64 t1 = [self getCurrentTimeNow];
    
    id<MTLCommandBuffer> commandBuffer = [_queue commandBuffer];
    [_net computeWithCommandBuffer:commandBuffer
                             input:float16Data
                       inputLength:28 * 28 * 4 * 2
                            output:float16Output
                      outputLength:10 * 2];
    
    UInt64 t2 = [self getCurrentTimeNow];
    
    convertFloat16ToFloat32((uint16_t*)float16Output, (uint32_t*)realOutput, 10);
    
    int max = 0;
    for (int k = 0; k < 10; ++k) {
        if (realOutput[k] > realOutput[max]) {
            max = k;
        }
    }
    
    free(rawData);
    free(float16Data);
    free(float16Output);
    free(realOutput);
    
    NSString *text = [NSString stringWithFormat:@"the digit number is:%d, cost time:%llu ms", max, (t2  -t1)];
    
    _result.text = text;
    [_result sizeToFit];
}

- (void)buildLeNet {
    /**load the LeNet model to test the Brouhaha*/
    NSBundle *mainBundle = [NSBundle mainBundle];
    
    _device = MTLCreateSystemDefaultDevice();
    _queue = [_device newCommandQueue];
    
    NSError *libraryError = NULL;
    NSString *libraryFile = [mainBundle pathForResource:@"BrouhahaMetal" ofType:@"metallib"];
    id <MTLLibrary> library = [_device newLibraryWithFile:libraryFile error:&libraryError];
    
    /**
     * read the LeNet model from file
     * the model file is from Internet, but I forget the source...
     */
    void *w0 = [self readBinaryFile:[mainBundle pathForResource:@"w0" ofType:@""] length:20*1*5*5*4];
    void *b0 = [self readBinaryFile:[mainBundle pathForResource:@"b0" ofType:@""] length:20*4];
    void *w1 = [self readBinaryFile:[mainBundle pathForResource:@"w1" ofType:@""] length:50*20*5*5*4];
    void *b1 = [self readBinaryFile:[mainBundle pathForResource:@"b1" ofType:@""] length:50*4];
    void *w2 = [self readBinaryFile:[mainBundle pathForResource:@"w2" ofType:@""] length:500*800*4];
    void *b2 = [self readBinaryFile:[mainBundle pathForResource:@"b2" ofType:@""] length:500*4];
    void *w3 = [self readBinaryFile:[mainBundle pathForResource:@"w3" ofType:@""] length:10*500*4];
    void *b3 = [self readBinaryFile:[mainBundle pathForResource:@"b3" ofType:@""] length:10*4];
    
    /**create LeNet layer*/
    BrouConvolutionMMLayer *con0 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:_device
                                                                                 library:library
                                                                                  kernel:w0
                                                                                    bias:nil
                                                                             inputHeight:28
                                                                              inputWidth:28
                                                                           intputChannel:1
                                                                            outputHeight:24
                                                                             outputWidth:24
                                                                           outputChannel:20
                                                                            kernelHeight:5
                                                                             kernelWidth:5
                                                                                 padLeft:0
                                                                                  padTop:0
                                                                                 strideX:1
                                                                                 strideY:1];
    
    BrouMaxPoolingLayer *maxPooling0 = [[BrouMaxPoolingLayer alloc] initWithDevice:_device
                                                                           library:library
                                                                       inputHeight:24
                                                                        inputWidth:24
                                                                      outputHeight:12
                                                                       outputWidth:12
                                                                           channel:20
                                                                      kernelHeight:2
                                                                       kernelWidth:2
                                                                           padLeft:0
                                                                            padTop:0
                                                                           strideX:2
                                                                           strideY:2];
    
    NSArray<NSNumber*> *add0BiasDim = @[@(1), @(1), @(20)];
    NSArray<NSNumber*> *add0InDim   = @[@(12), @(12), @(20)];
    
    BrouAddLayer *add0 = [[BrouAddLayer alloc] initWithDevice:_device
                                                      library:library
                                                  float32Bias:b0
                                                      biasDim:add0BiasDim
                                                        inDim:add0InDim];
    
    BrouTanHLayer *tanh0 = [[BrouTanHLayer alloc] initWithDevice:_device
                                                         library:library
                                                          height:12
                                                           width:12
                                                         channel:20];
    
    BrouConvolutionMMLayer *con1 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:_device
                                                                                 library:library
                                                                                  kernel:w1
                                                                                    bias:nil
                                                                             inputHeight:12
                                                                              inputWidth:12
                                                                           intputChannel:20
                                                                            outputHeight:8
                                                                             outputWidth:8
                                                                           outputChannel:50
                                                                            kernelHeight:5
                                                                             kernelWidth:5
                                                                                 padLeft:0
                                                                                  padTop:0
                                                                                 strideX:1
                                                                                 strideY:1];
    
    BrouMaxPoolingLayer *maxPooling1 = [[BrouMaxPoolingLayer alloc] initWithDevice:_device
                                                                           library:library
                                                                       inputHeight:8
                                                                        inputWidth:8
                                                                      outputHeight:4
                                                                       outputWidth:4
                                                                           channel:50
                                                                      kernelHeight:2
                                                                       kernelWidth:2
                                                                           padLeft:0
                                                                            padTop:0
                                                                           strideX:2
                                                                           strideY:2];
    
    NSArray<NSNumber*> *add1BiasDim = @[@(1), @(1), @(50)];
    NSArray<NSNumber*> *add1InDim   = @[@(2), @(2), @(50)];
    
    BrouAddLayer *add1 = [[BrouAddLayer alloc] initWithDevice:_device
                                                      library:library
                                                  float32Bias:b1
                                                      biasDim:add1BiasDim
                                                        inDim:add1InDim];
    
    BrouTanHLayer *tanh1 = [[BrouTanHLayer alloc] initWithDevice:_device
                                                         library:library
                                                          height:2
                                                           width:2
                                                         channel:50];
    
    BrouConvolutionMMLayer *fullConnect0 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:_device
                                                                                         library:library
                                                                                          kernel:w2
                                                                                            bias:b2
                                                                                     inputHeight:4
                                                                                      inputWidth:4
                                                                                   intputChannel:50
                                                                                    outputHeight:1
                                                                                     outputWidth:1
                                                                                   outputChannel:500
                                                                                    kernelHeight:4
                                                                                     kernelWidth:4
                                                                                         padLeft:0
                                                                                          padTop:0
                                                                                         strideX:0
                                                                                         strideY:0];
    
    BrouTanHLayer *tanh2 = [[BrouTanHLayer alloc] initWithDevice:_device library:library channel:500];
    
    BrouFullConnectLayer *fullConnect1 = [[BrouFullConnectLayer alloc] initWithFloat32Device:_device
                                                                                     library:library
                                                                                     weights:w3
                                                                                        bias:b3
                                                                               intputChannel:500
                                                                               outputChannel:10];
    
    _net = [[BrouNet alloc] init];
    [_net addLayer:con0];
    [_net addLayer:maxPooling0];
    [_net addLayer:add0];
    [_net addLayer:tanh0];
    [_net addLayer:con1];
    [_net addLayer:maxPooling1];
    [_net addLayer:add1];
    [_net addLayer:tanh1];
    [_net addLayer:fullConnect0];
    [_net addLayer:tanh2];
    [_net addLayer:fullConnect1];
    
    /**config the net*/
    [_net configWithDevice:_device];
}

- (void)click:(id)sender {
    if (_closeButton == sender) {
        [self dismissViewControllerAnimated:YES completion:nil];
    } else if (_recognizeButton == sender) {
        [self recognizeDigitNumber];
    } else if (_clearButton == sender) {
        [_paintView clear];
    }
}

- (UInt64)getCurrentTimeNow {
    return [[NSDate date] timeIntervalSince1970] * 1000;
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
