/**
 * Created by yanyuanchi on 2017/7/23.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the art transform viewcontoller
 */
#import "sys/mman.h"

#import "ArtTransformViewController.h"
#import "BrouConvolutionLayer.h"
#import "BrouConvolutionMMLayer.h"
#import "BrouMaxPoolingLayer.h"
#import "BrouFullConnectLayer.h"
#import "BrouTanHLayer.h"
#import "BrouAddLayer.h"
#import "BrouNet.h"
#import "BrouUtils.h"
#import "BrouAddLayer.h"
#import "BrouBatchNormalizationLayer.h"
#import "BrouReLuLayer.h"
#import "BrouResidualLayer.h"
#import "BrouTransposedConvolutionLayer.h"
#import "BrouTransposedConvolutionMMLayer.h"
#import "BrouRGBAConvertLayer.h"
#import "BrouLinearLayer.h"
#import "BrouNeon.h"

/**
 * the art transform ref:https://github.com/lengstrom/fast-style-transfer#video-stylization
 * the params of convolution network is form:https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?usp=sharing
 */
@interface ArtTransformViewController ()

@property(nonatomic, strong) UIButton *button;

@property(nonatomic, strong) UIButton *closeButton;

@property(nonatomic, strong) UIImageView *imageView;

@property(nonatomic, strong) UIImage *originImage;
@property(nonatomic, strong) UIImage *artImage;

@property(nonatomic, assign) BOOL isArt;

@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;

@property(nonatomic, strong) BrouNet *net;

@end

@implementation ArtTransformViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    NSBundle *mainBundle = [NSBundle mainBundle];
    NSString *imagePath  = [mainBundle pathForResource:@"zgr1" ofType:@"jpg"];
    _originImage = [UIImage imageWithContentsOfFile:imagePath];
    
    int width  = (int)CGImageGetWidth(_originImage.CGImage);
    int height = (int)CGImageGetHeight(_originImage.CGImage);
    
    _device = MTLCreateSystemDefaultDevice();
    _queue  = [_device newCommandQueue];
    
    NSError *libraryError   = NULL;
    NSString *libraryFile   = [mainBundle pathForResource:@"BrouhahaMetal" ofType:@"metallib"];
    _library = [_device newLibraryWithFile:libraryFile error:&libraryError];
    
    [self buildNetWithHeight:height width:width device:_device library:_library];
    
    CGRect bounds = self.view.bounds;
    
    /**init views*/
    _imageView = [[UIImageView alloc] init];
    _imageView.contentMode = UIViewContentModeScaleAspectFit;
    _imageView.frame = bounds;
    [_imageView setImage:_originImage];
    
    _button = [UIButton buttonWithType:UIButtonTypeSystem];
    [_button setTitle:@"Click me! Click me!" forState:UIControlStateNormal];
    [_button setFrame:CGRectMake(0, bounds.size.height - 100, bounds.size.width, 50)];
    [_button addTarget:self action:@selector(click:) forControlEvents:UIControlEventTouchUpInside];
    
    _closeButton = [UIButton buttonWithType:UIButtonTypeSystem];
    [_closeButton setTitle:@"close" forState:UIControlStateNormal];
    [_closeButton setFrame:CGRectMake(0, bounds.size.height - 100, 100, 50)];
    [_closeButton addTarget:self action:@selector(closeClick:) forControlEvents:UIControlEventTouchUpInside];
    
    [self.view addSubview:_imageView];
    [self.view addSubview:_button];
    [self.view addSubview:_closeButton];
    
    self.view.backgroundColor = [UIColor whiteColor];
    
    _isArt = NO;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

- (void)closeClick:(id)sender {
    [self dismissViewControllerAnimated:YES completion:nil];
}

- (void)click:(id)sender{
    if (_isArt) {
        [_imageView setImage:_originImage];
        
        _isArt = NO;
    } else {
        if (!_artImage) {
            int width  = (int)CGImageGetWidth(_originImage.CGImage);
            int height = (int)CGImageGetHeight(_originImage.CGImage);
            
            void *inputRawData = [self getPixelsFromImage:_originImage.CGImage];
            void *outputRawData = malloc(height * width * 4);
            
            id<MTLCommandBuffer> commandBuffer = [_queue commandBuffer];
            
            [_net computeWithCommandBuffer:commandBuffer
                                     input:inputRawData
                               inputLength:height * width * 4
                                    output:outputRawData
                              outputLength:height * width * 4];
            
            _artImage = [self getUIImageFromPixels:outputRawData width:width height:height];
            
            free(inputRawData);
        }
        
        [_imageView setImage:_artImage];
        
        _isArt = YES;
    }
}

- (void)buildNetWithHeight:(int)height width:(int)width device:(id<MTLDevice>)device library:(id<MTLLibrary>)library {
    NSBundle *mainBundle = [NSBundle mainBundle];
    
    void *conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv1_weight" ofType:@""] length:32*9*9*3*4];
    void *conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv1_alpha" ofType:@""] length:32*4];
    void *conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv1_beta" ofType:@""] length:32*4];
    
    void *conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv2_weight" ofType:@""] length:64*3*3*32*4];
    void *conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv2_alpha" ofType:@""] length:64*4];
    void *conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv2_beta" ofType:@""] length:64*4];
    
    void *conv3_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv3_weight" ofType:@""] length:128*3*3*64*4];
    void *conv3_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv3_alpha" ofType:@""] length:128*4];
    void *conv3_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv3_beta" ofType:@""] length:128*4];
    
    void *res1_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res1_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv1_alpha" ofType:@""] length:128*4];
    void *res1_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv1_beta" ofType:@""] length:128*4];
    void *res1_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res1_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv2_alpha" ofType:@""] length:128*4];
    void *res1_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv2_beta" ofType:@""] length:128*4];
    
    void *res2_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res2_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv1_alpha" ofType:@""] length:128*4];
    void *res2_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv1_beta" ofType:@""] length:128*4];
    void *res2_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res2_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv2_alpha" ofType:@""] length:128*4];
    void *res2_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv2_beta" ofType:@""] length:128*4];
    
    void *res3_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res3_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv1_alpha" ofType:@""] length:128*4];
    void *res3_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv1_beta" ofType:@""] length:128*4];
    void *res3_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res3_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv2_alpha" ofType:@""] length:128*4];
    void *res3_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv2_beta" ofType:@""] length:128*4];
    
    void *res4_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res4_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv1_alpha" ofType:@""] length:128*4];
    void *res4_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv1_beta" ofType:@""] length:128*4];
    void *res4_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res4_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv2_alpha" ofType:@""] length:128*4];
    void *res4_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv2_beta" ofType:@""] length:128*4];
    
    void *res5_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res5_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv1_alpha" ofType:@""] length:128*4];
    void *res5_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv1_beta" ofType:@""] length:128*4];
    void *res5_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res5_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv2_alpha" ofType:@""] length:128*4];
    void *res5_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv2_beta" ofType:@""] length:128*4];
    
    void *transpose_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv1_weight" ofType:@""] length:128*64*3*3*4];
    void *transpose_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv1_alpha" ofType:@""] length:64*4];
    void *transpose_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv1_beta" ofType:@""] length:64*4];
    
    void *transpose_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv2_weight" ofType:@""] length:64*32*3*3*4];
    void *transpose_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv2_alpha" ofType:@""] length:32*4];
    void *transpose_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv2_beta" ofType:@""] length:32*4];
    
    void *conv4_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv4_weight" ofType:@""] length:3*9*9*32*4];
    void *conv4_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv4_alpha" ofType:@""] length:3*4];
    void *conv4_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv4_beta" ofType:@""] length:3*4];
    
    float epsilon = 0.001;
    
    int inHeight = height;
    int inWidth  = width;
    int inChannel = 3;
    
    int outHeight = height;
    int outWidth  = width;
    int outChannel = 32;
    
    BrouRGBAConvertLayer *rgbaConvert1 = [[BrouRGBAConvertLayer alloc] initWithDevice:device
                                                                              library:library
                                                                               height:height
                                                                                width:width
                                                                          convertType:UINT8_TO_HALF];
    
    BrouConvolutionMMLayer *conv1 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                                  library:library
                                                                                   kernel:conv1_weight
                                                                                     bias:nil
                                                                              inputHeight:inHeight
                                                                               inputWidth:inWidth
                                                                            intputChannel:inChannel
                                                                             outputHeight:outHeight
                                                                              outputWidth:outWidth
                                                                            outputChannel:outChannel
                                                                             kernelHeight:9
                                                                              kernelWidth:9
                                                                                  padLeft:4
                                                                                   padTop:4
                                                                                  strideX:1
                                                                                  strideY:1];
    
    BrouBatchNormalizationLayer *batchNorm1 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv1_alpha
                                                                                                    beta:conv1_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu1 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight = outHeight;
    inWidth  = outWidth;
    
    inChannel = outChannel;
    
    outHeight = ceil(1.0 * inHeight / 2.0);
    outWidth  = ceil(1.0 * inWidth / 2.0);
    
    outChannel = 64;
    
    BrouConvolutionMMLayer *conv2 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                                  library:library
                                                                                   kernel:conv2_weight
                                                                                     bias:nil
                                                                              inputHeight:inHeight
                                                                               inputWidth:inWidth
                                                                            intputChannel:inChannel
                                                                             outputHeight:outHeight
                                                                              outputWidth:outWidth
                                                                            outputChannel:outChannel
                                                                             kernelHeight:3
                                                                              kernelWidth:3
                                                                                  padLeft:0
                                                                                   padTop:0
                                                                                  strideX:2
                                                                                  strideY:2];
    
    BrouBatchNormalizationLayer *batchNorm2 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv2_alpha
                                                                                                    beta:conv2_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu2 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = ceil(1.0 * inHeight / 2.0);
    outWidth  = ceil(1.0 * inWidth / 2.0);
    
    outChannel = 128;
    
    BrouConvolutionMMLayer *conv3 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                                  library:library
                                                                                   kernel:conv3_weight
                                                                                     bias:nil
                                                                              inputHeight:inHeight
                                                                               inputWidth:inWidth
                                                                            intputChannel:inChannel
                                                                             outputHeight:outHeight
                                                                              outputWidth:outWidth
                                                                            outputChannel:outChannel
                                                                             kernelHeight:3
                                                                              kernelWidth:3
                                                                                  padLeft:0
                                                                                   padTop:0
                                                                                  strideX:2
                                                                                  strideY:2];
    
    BrouBatchNormalizationLayer *batchNorm3 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv3_alpha
                                                                                                    beta:conv3_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu3 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    BrouResidualLayer *res1 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res1_conv1_weight
                                                         float32Weight2:res1_conv2_weight
                                                          float32Alpha1:res1_conv1_alpha
                                                           float32Beta1:res1_conv1_beta
                                                          float32Alpha2:res1_conv2_alpha
                                                           float32Beta2:res1_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res2 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res2_conv1_weight
                                                         float32Weight2:res2_conv2_weight
                                                          float32Alpha1:res2_conv1_alpha
                                                           float32Beta1:res2_conv1_beta
                                                          float32Alpha2:res2_conv2_alpha
                                                           float32Beta2:res2_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res3 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res3_conv1_weight
                                                         float32Weight2:res3_conv2_weight
                                                          float32Alpha1:res3_conv1_alpha
                                                           float32Beta1:res3_conv1_beta
                                                          float32Alpha2:res3_conv2_alpha
                                                           float32Beta2:res3_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res4 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res4_conv1_weight
                                                         float32Weight2:res4_conv2_weight
                                                          float32Alpha1:res4_conv1_alpha
                                                           float32Beta1:res4_conv1_beta
                                                          float32Alpha2:res4_conv2_alpha
                                                           float32Beta2:res4_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res5 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res5_conv1_weight
                                                         float32Weight2:res5_conv2_weight
                                                          float32Alpha1:res5_conv1_alpha
                                                           float32Beta1:res5_conv1_beta
                                                          float32Alpha2:res5_conv2_alpha
                                                           float32Beta2:res5_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    inHeight = outHeight;
    inWidth  = outWidth;
    inChannel = outChannel;
    
    outHeight = 2 * inHeight;
    outWidth  = 2 * inWidth;
    outChannel = 64;
    
    BrouTransposedConvolutionMMLayer *transpose_conv1 = [[BrouTransposedConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                                                                library:library
                                                                                                                 kernel:transpose_conv1_weight
                                                                                                                   bias:nil
                                                                                                      originInputHeight:outHeight
                                                                                                       originInputWidth:outWidth
                                                                                                     originInputChannel:outChannel
                                                                                                     originOutputHeight:inHeight
                                                                                                      originOutputWidth:inWidth
                                                                                                    originOutputChannel:inChannel
                                                                                                     originKernelHeight:3
                                                                                                      originKernelWidth:3
                                                                                                          originPadLeft:0
                                                                                                           originPadTop:0
                                                                                                          originStrideX:2
                                                                                                          originStrideY:2
                                                                                                         outputAddRight:0
                                                                                                        outputAddBottom:0];
    
    BrouBatchNormalizationLayer *batchNorm4 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:transpose_conv1_alpha
                                                                                                    beta:transpose_conv1_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu4 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = 2 * inHeight;
    outWidth  = 2 * inWidth;
    outChannel = 32;
    
    BrouTransposedConvolutionMMLayer *transpose_conv2 = [[BrouTransposedConvolutionMMLayer alloc]
                                                         initWithFloat32Device:device
                                                         library:library
                                                         kernel:transpose_conv2_weight
                                                         bias:nil
                                                         originInputHeight:outHeight
                                                         originInputWidth:outWidth
                                                         originInputChannel:outChannel
                                                         originOutputHeight:inHeight
                                                         originOutputWidth:inWidth
                                                         originOutputChannel:inChannel
                                                         originKernelHeight:3
                                                         originKernelWidth:3
                                                         originPadLeft:0
                                                         originPadTop:0
                                                         originStrideX:2
                                                         originStrideY:2
                                                         outputAddRight:0
                                                         outputAddBottom:0];
    
    BrouBatchNormalizationLayer *batchNorm5 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:transpose_conv2_alpha
                                                                                                    beta:transpose_conv2_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu5 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = inHeight;
    outWidth  = inWidth;
    outChannel = 3;
    
    BrouConvolutionLayer *conv4 = [[BrouConvolutionLayer alloc] initWithFloat32Device:device
                                                                              library:library
                                                                               kernel:conv4_weight
                                                                                 bias:nil
                                                                          inputHeight:inHeight
                                                                           inputWidth:inWidth
                                                                        intputChannel:inChannel
                                                                         outputHeight:outHeight
                                                                          outputWidth:outWidth
                                                                        outputChannel:outChannel
                                                                         kernelHeight:9
                                                                          kernelWidth:9
                                                                              padLeft:4
                                                                               padTop:4
                                                                              strideX:1
                                                                              strideY:1];
    
    BrouBatchNormalizationLayer *batchNorm6 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv4_alpha
                                                                                                    beta:conv4_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = inHeight;
    outWidth  = inWidth;
    outChannel = inChannel;
    
    BrouTanHLayer *tanh1 = [[BrouTanHLayer alloc] initWithDevice:device
                                                         library:library
                                                          height:outHeight
                                                           width:outWidth
                                                         channel:outChannel];
    
    BrouLinearLayer *linear = [[BrouLinearLayer alloc] initWithDevice:device
                                                              library:library
                                                               height:outHeight
                                                                width:outWidth
                                                              channel:outChannel
                                                                    a:150
                                                                    b:255./2];
    
    BrouRGBAConvertLayer *rgbaConvert2 = [[BrouRGBAConvertLayer alloc] initWithDevice:device
                                                                              library:library
                                                                               height:height
                                                                                width:width
                                                                          convertType:HALF_TO_UINT8];
    
    _net = [[BrouNet alloc] init];
    [_net addLayer:rgbaConvert1];
    
    [_net addLayer:conv1];
    [_net addLayer:batchNorm1];
    [_net addLayer:relu1];
    
    [_net addLayer:conv2];
    [_net addLayer:batchNorm2];
    [_net addLayer:relu2];
    
    [_net addLayer:conv3];
    [_net addLayer:batchNorm3];
    [_net addLayer:relu3];
    
    [_net addLayer:res1];
    [_net addLayer:res2];
    [_net addLayer:res3];
    [_net addLayer:res4];
    [_net addLayer:res5];
    
    [_net addLayer:transpose_conv1];
    [_net addLayer:batchNorm4];
    [_net addLayer:relu4];
    
    [_net addLayer:transpose_conv2];
    [_net addLayer:batchNorm5];
    [_net addLayer:relu5];
    
    [_net addLayer:conv4];
    [_net addLayer:batchNorm6];
    
    [_net addLayer:tanh1];
    [_net addLayer:linear];
    
    [_net addLayer:rgbaConvert2];
    
    /**config*/
    [_net configWithDevice:device];
}
- (void)artTransformTestWithImage:(UIImage*)image
                           device:(id<MTLDevice>)device
                          library:(id<MTLLibrary>)library
                            queue:(id<MTLCommandQueue>)queue {
    
    CGImageRef cgImage = image.CGImage;
    
    int width  = CGImageGetWidth(cgImage);
    int height = CGImageGetHeight(cgImage);
    
    NSBundle *mainBundle = [NSBundle mainBundle];
    
    void *conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv1_weight" ofType:@""] length:32*9*9*3*4];
    void *conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv1_alpha" ofType:@""] length:32*4];
    void *conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv1_beta" ofType:@""] length:32*4];

    void *conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv2_weight" ofType:@""] length:64*3*3*32*4];
    void *conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv2_alpha" ofType:@""] length:64*4];
    void *conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv2_beta" ofType:@""] length:64*4];
    
    void *conv3_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv3_weight" ofType:@""] length:128*3*3*64*4];
    void *conv3_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv3_alpha" ofType:@""] length:128*4];
    void *conv3_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv3_beta" ofType:@""] length:128*4];
    
    void *res1_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res1_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv1_alpha" ofType:@""] length:128*4];
    void *res1_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv1_beta" ofType:@""] length:128*4];
    void *res1_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res1_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv2_alpha" ofType:@""] length:128*4];
    void *res1_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res1_conv2_beta" ofType:@""] length:128*4];
    
    void *res2_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res2_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv1_alpha" ofType:@""] length:128*4];
    void *res2_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv1_beta" ofType:@""] length:128*4];
    void *res2_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res2_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv2_alpha" ofType:@""] length:128*4];
    void *res2_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res2_conv2_beta" ofType:@""] length:128*4];
    
    void *res3_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res3_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv1_alpha" ofType:@""] length:128*4];
    void *res3_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv1_beta" ofType:@""] length:128*4];
    void *res3_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res3_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv2_alpha" ofType:@""] length:128*4];
    void *res3_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res3_conv2_beta" ofType:@""] length:128*4];
    
    void *res4_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res4_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv1_alpha" ofType:@""] length:128*4];
    void *res4_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv1_beta" ofType:@""] length:128*4];
    void *res4_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res4_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv2_alpha" ofType:@""] length:128*4];
    void *res4_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res4_conv2_beta" ofType:@""] length:128*4];
    
    void *res5_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv1_weight" ofType:@""] length:128*3*3*128*4];
    void *res5_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv1_alpha" ofType:@""] length:128*4];
    void *res5_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv1_beta" ofType:@""] length:128*4];
    void *res5_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv2_weight" ofType:@""] length:128*3*3*128*4];
    void *res5_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv2_alpha" ofType:@""] length:128*4];
    void *res5_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"res5_conv2_beta" ofType:@""] length:128*4];
    
    void *transpose_conv1_weight = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv1_weight" ofType:@""] length:128*64*3*3*4];
    void *transpose_conv1_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv1_alpha" ofType:@""] length:64*4];
    void *transpose_conv1_beta   = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv1_beta" ofType:@""] length:64*4];
    
    void *transpose_conv2_weight = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv2_weight" ofType:@""] length:64*32*3*3*4];
    void *transpose_conv2_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv2_alpha" ofType:@""] length:32*4];
    void *transpose_conv2_beta   = [self readBinaryFile:[mainBundle pathForResource:@"transpose_conv2_beta" ofType:@""] length:32*4];
    
    void *conv4_weight = [self readBinaryFile:[mainBundle pathForResource:@"conv4_weight" ofType:@""] length:3*9*9*32*4];
    void *conv4_alpha  = [self readBinaryFile:[mainBundle pathForResource:@"conv4_alpha" ofType:@""] length:3*4];
    void *conv4_beta   = [self readBinaryFile:[mainBundle pathForResource:@"conv4_beta" ofType:@""] length:3*4];

    float epsilon = 0.001;
    
    int inHeight = height;
    int inWidth  = width;
    int inChannel = 3;
    
    int outHeight = height;
    int outWidth  = width;
    int outChannel = 32;
    
    BrouRGBAConvertLayer *rgbaConvert1 = [[BrouRGBAConvertLayer alloc] initWithDevice:device
                                                                              library:library
                                                                               height:height
                                                                                width:width
                                                                          convertType:UINT8_TO_HALF];
    
    BrouConvolutionMMLayer *conv1 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                              library:library
                                                                               kernel:conv1_weight
                                                                                 bias:nil
                                                                          inputHeight:inHeight
                                                                           inputWidth:inWidth
                                                                        intputChannel:inChannel
                                                                         outputHeight:outHeight
                                                                          outputWidth:outWidth
                                                                        outputChannel:outChannel
                                                                         kernelHeight:9
                                                                          kernelWidth:9
                                                                              padLeft:4
                                                                               padTop:4
                                                                              strideX:1
                                                                              strideY:1];
    
    BrouBatchNormalizationLayer *batchNorm1 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv1_alpha
                                                                                                    beta:conv1_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu1 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight = outHeight;
    inWidth  = outWidth;
    
    inChannel = outChannel;
    
    outHeight = ceil(1.0 * inHeight / 2.0);
    outWidth  = ceil(1.0 * inWidth / 2.0);
    
    outChannel = 64;
    
    BrouConvolutionMMLayer *conv2 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                              library:library
                                                                               kernel:conv2_weight
                                                                                 bias:nil
                                                                          inputHeight:inHeight
                                                                           inputWidth:inWidth
                                                                        intputChannel:inChannel
                                                                         outputHeight:outHeight
                                                                          outputWidth:outWidth
                                                                        outputChannel:outChannel
                                                                         kernelHeight:3
                                                                          kernelWidth:3
                                                                              padLeft:0
                                                                               padTop:0
                                                                              strideX:2
                                                                              strideY:2];
    
    BrouBatchNormalizationLayer *batchNorm2 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv2_alpha
                                                                                                    beta:conv2_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu2 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = ceil(1.0 * inHeight / 2.0);
    outWidth  = ceil(1.0 * inWidth / 2.0);
    
    outChannel = 128;
    
    BrouConvolutionMMLayer *conv3 = [[BrouConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                              library:library
                                                                               kernel:conv3_weight
                                                                                 bias:nil
                                                                          inputHeight:inHeight
                                                                           inputWidth:inWidth
                                                                        intputChannel:inChannel
                                                                         outputHeight:outHeight
                                                                          outputWidth:outWidth
                                                                        outputChannel:outChannel
                                                                         kernelHeight:3
                                                                          kernelWidth:3
                                                                              padLeft:0
                                                                               padTop:0
                                                                              strideX:2
                                                                              strideY:2];
    
    BrouBatchNormalizationLayer *batchNorm3 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv3_alpha
                                                                                                    beta:conv3_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu3 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    BrouResidualLayer *res1 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res1_conv1_weight
                                                         float32Weight2:res1_conv2_weight
                                                          float32Alpha1:res1_conv1_alpha
                                                           float32Beta1:res1_conv1_beta
                                                          float32Alpha2:res1_conv2_alpha
                                                           float32Beta2:res1_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res2 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res2_conv1_weight
                                                         float32Weight2:res2_conv2_weight
                                                          float32Alpha1:res2_conv1_alpha
                                                           float32Beta1:res2_conv1_beta
                                                          float32Alpha2:res2_conv2_alpha
                                                           float32Beta2:res2_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res3 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res3_conv1_weight
                                                         float32Weight2:res3_conv2_weight
                                                          float32Alpha1:res3_conv1_alpha
                                                           float32Beta1:res3_conv1_beta
                                                          float32Alpha2:res3_conv2_alpha
                                                           float32Beta2:res3_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res4 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res4_conv1_weight
                                                         float32Weight2:res4_conv2_weight
                                                          float32Alpha1:res4_conv1_alpha
                                                           float32Beta1:res4_conv1_beta
                                                          float32Alpha2:res4_conv2_alpha
                                                           float32Beta2:res4_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    BrouResidualLayer *res5 = [[BrouResidualLayer alloc] initWithDevice:device
                                                                library:library
                                                         float32Weight1:res5_conv1_weight
                                                         float32Weight2:res5_conv2_weight
                                                          float32Alpha1:res5_conv1_alpha
                                                           float32Beta1:res5_conv1_beta
                                                          float32Alpha2:res5_conv2_alpha
                                                           float32Beta2:res5_conv2_beta
                                                                 height:outHeight
                                                                  width:outWidth];
    
    inHeight = outHeight;
    inWidth  = outWidth;
    inChannel = outChannel;
    
    outHeight = 2 * inHeight;
    outWidth  = 2 * inWidth;
    outChannel = 64;
    
    BrouTransposedConvolutionMMLayer *transpose_conv1 = [[BrouTransposedConvolutionMMLayer alloc] initWithFloat32Device:device
                                                                                                            library:library
                                                                                                             kernel:transpose_conv1_weight
                                                                                                               bias:nil
                                                                                                  originInputHeight:outHeight
                                                                                                   originInputWidth:outWidth
                                                                                                 originInputChannel:outChannel
                                                                                                 originOutputHeight:inHeight
                                                                                                  originOutputWidth:inWidth
                                                                                                originOutputChannel:inChannel
                                                                                                 originKernelHeight:3
                                                                                                  originKernelWidth:3
                                                                                                      originPadLeft:0
                                                                                                       originPadTop:0
                                                                                                      originStrideX:2
                                                                                                      originStrideY:2
                                                                                                     outputAddRight:0
                                                                                                    outputAddBottom:0];
    
    BrouBatchNormalizationLayer *batchNorm4 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:transpose_conv1_alpha
                                                                                                    beta:transpose_conv1_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu4 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = 2 * inHeight;
    outWidth  = 2 * inWidth;
    outChannel = 32;
    
    BrouTransposedConvolutionMMLayer *transpose_conv2 = [[BrouTransposedConvolutionMMLayer alloc]
                                                       initWithFloat32Device:device
                                                       library:library
                                                       kernel:transpose_conv2_weight
                                                       bias:nil
                                                       originInputHeight:outHeight
                                                       originInputWidth:outWidth
                                                       originInputChannel:outChannel
                                                       originOutputHeight:inHeight
                                                       originOutputWidth:inWidth
                                                       originOutputChannel:inChannel
                                                       originKernelHeight:3
                                                       originKernelWidth:3
                                                       originPadLeft:0
                                                       originPadTop:0
                                                       originStrideX:2
                                                       originStrideY:2
                                                       outputAddRight:0
                                                       outputAddBottom:0];
    
    BrouBatchNormalizationLayer *batchNorm5 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:transpose_conv2_alpha
                                                                                                    beta:transpose_conv2_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    BrouReLuLayer *relu5 = [[BrouReLuLayer alloc] initReLuWithDevice:device
                                                             library:library
                                                              height:outHeight
                                                               width:outWidth
                                                             channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = inHeight;
    outWidth  = inWidth;
    outChannel = 3;
    
    BrouConvolutionLayer *conv4 = [[BrouConvolutionLayer alloc] initWithFloat32Device:device
                                                                              library:library
                                                                               kernel:conv4_weight
                                                                                 bias:nil
                                                                          inputHeight:inHeight
                                                                           inputWidth:inWidth
                                                                        intputChannel:inChannel
                                                                         outputHeight:outHeight
                                                                          outputWidth:outWidth
                                                                        outputChannel:outChannel
                                                                         kernelHeight:9
                                                                          kernelWidth:9
                                                                              padLeft:4
                                                                               padTop:4
                                                                              strideX:1
                                                                              strideY:1];
    
    BrouBatchNormalizationLayer *batchNorm6 = [[BrouBatchNormalizationLayer alloc] initWithFloat32Device:device
                                                                                                 library:library
                                                                                                   alpha:conv4_alpha
                                                                                                    beta:conv4_beta
                                                                                                 epsilon:epsilon
                                                                                                  height:outHeight
                                                                                                   width:outWidth
                                                                                                 channel:outChannel];
    
    inHeight  = outHeight;
    inWidth   = outWidth;
    inChannel = outChannel;
    
    outHeight = inHeight;
    outWidth  = inWidth;
    outChannel = inChannel;
    
    BrouTanHLayer *tanh1 = [[BrouTanHLayer alloc] initWithDevice:device
                                                         library:library
                                                          height:outHeight
                                                           width:outWidth
                                                         channel:outChannel];
    
    BrouLinearLayer *linear = [[BrouLinearLayer alloc] initWithDevice:device
                                                              library:library
                                                               height:outHeight
                                                                width:outWidth
                                                              channel:outChannel
                                                                    a:150
                                                                    b:255./2];
    
    BrouRGBAConvertLayer *rgbaConvert2 = [[BrouRGBAConvertLayer alloc] initWithDevice:device
                                                                              library:library
                                                                               height:height
                                                                                width:width
                                                                          convertType:HALF_TO_UINT8];
    
    BrouNet *net = [[BrouNet alloc] init];
    [net addLayer:rgbaConvert1];

    [net addLayer:conv1];
    [net addLayer:batchNorm1];
    [net addLayer:relu1];

    [net addLayer:conv2];
    [net addLayer:batchNorm2];
    [net addLayer:relu2];

    [net addLayer:conv3];
    [net addLayer:batchNorm3];
    [net addLayer:relu3];

    [net addLayer:res1];
    [net addLayer:res2];
    [net addLayer:res3];
    [net addLayer:res4];
    [net addLayer:res5];

    [net addLayer:transpose_conv1];
    [net addLayer:batchNorm4];
    [net addLayer:relu4];

    [net addLayer:transpose_conv2];
    [net addLayer:batchNorm5];
    [net addLayer:relu5];

    [net addLayer:conv4];
    [net addLayer:batchNorm6];

    [net addLayer:tanh1];
    [net addLayer:linear];

    [net addLayer:rgbaConvert2];

    /**config*/
    [net configWithDevice:device];
    
    void *inputRawData = [self getPixelsFromImage:cgImage];
    void *outputRawData = malloc(height * width * 4);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    
    [net computeWithCommandBuffer:commandBuffer
                            input:inputRawData
                      inputLength:height * width * 4
                           output:outputRawData
                     outputLength:height * width * 4];
    
    UIImage *artImage = [self getUIImageFromPixels:outputRawData width:width height:height];
    
    NSLog(@"yyc");
}

/**
 * read binary file to memory
 */
- (void*)readBinaryFile:(NSString*)path length:(int)length {
    int file = open([path UTF8String], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    
    assert(file != -1);
    
    void * filePointer = mmap(nil, length, PROT_READ, MAP_FILE | MAP_SHARED, file, 0);
    
    NSAssert(MAP_FAILED != filePointer, @"read file fail!");
    
    close(file);
    
    return filePointer;
}

/**
 * pixels data from a image
 * the data dimension is (height, width, [r,g,b,a])
 */
- (void*)getPixelsFromImage:(CGImageRef)image {
    NSUInteger width  = CGImageGetWidth(image);
    NSUInteger height = CGImageGetHeight(image);
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    
    void *rawData = malloc(height * width * 4);
    
    NSUInteger bytesPerRow = width * 4;
    NSUInteger bitsPerComponent = 8;
    
    CGContextRef context = CGBitmapContextCreate(rawData,
                                                 width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big);
    
    CGColorSpaceRelease(colorSpace);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    
    return rawData;
}

/**
 * form a RGBX pixels rawdata get a uiimage
 */
- (UIImage*)getUIImageFromPixels:(void*)rawData width:(NSInteger)width height:(NSInteger)height {
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL,
                                                              rawData,
                                                              width * height * 4,
                                                              NULL);
    
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGImageRef cgImage =  CGImageCreate(width, height,
                                        8, 32,
                                        width * 4,
                                        colorSpace,
                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Big,
                                        provider,
                                        nil,
                                        NO,
                                        kCGRenderingIntentDefault);
    
    UIImage *image = [UIImage imageWithCGImage:cgImage];
    
    return image;
}

@end

































