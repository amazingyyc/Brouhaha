#import "sys/mman.h"
#import "ArtTransformHalfViewController.h"
#import "Brouhaha.h"
#import "BrouResidualLayer_half.h"
#import "BrouTemporaryBuffer.h"

@interface ArtTransformHalfViewController () {
    BrouConvertFromuchar2halfLayer *rgbConvert1;
    
    BrouConvolutionLayer_half *conv1;
    BrouBatchNormalizationLayer_half *batchNorm1;
    BrouReLuLayer_half *relu1;
    
    BrouConvolutionMMLayer_half *conv2;
    BrouBatchNormalizationLayer_half *batchNorm2;
    BrouReLuLayer_half *relu2;
    
    BrouConvolutionMMLayer_half *conv3;
    BrouBatchNormalizationLayer_half *batchNorm3;
    BrouReLuLayer_half *relu3;
    
    BrouResidualLayer_half *res1;
    BrouResidualLayer_half *res2;
    BrouResidualLayer_half *res3;
    BrouResidualLayer_half *res4;
    BrouResidualLayer_half *res5;
    
    BrouTransposedConvolutionMMLayer_half *transposeConv1;
    
    BrouBatchNormalizationLayer_half *batchNorm4;
    BrouReLuLayer_half *relu4;
    
    BrouTransposedConvolutionMMLayer_half *transposeConv2;
    
    BrouBatchNormalizationLayer_half *batchNorm5;
    BrouReLuLayer_half *relu5;
    
    BrouConvolutionLayer_half *conv4;
    
    BrouBatchNormalizationLayer_half *batchNorm6;
    
    BrouTanHLayer_half *tanh1;
    BrouLinearLayer_half *linear1;
    
    BrouConvertFromhalf2ucharLayer *rgbConvert2;
    
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
    
    BrouTemporaryBuffer *buffer1;
    BrouTemporaryBuffer *buffer2;
}

@property(nonatomic, strong) UIButton *button;
@property(nonatomic, strong) UIButton *closeButton;
@property(nonatomic, strong) UIImageView *imageView;
@property(nonatomic, strong) UIImage *originImage;
@property(nonatomic, strong) UIImage *artImage;
@property(nonatomic, assign) BOOL isArt;

@end

@implementation ArtTransformHalfViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSBundle *mainBundle = [NSBundle mainBundle];
    NSString *imagePath  = [mainBundle pathForResource:@"tk" ofType:@"png"];
    _originImage = [UIImage imageWithContentsOfFile:imagePath];
    
    /**init views*/
    CGRect bounds = self.view.bounds;
    
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
    
    [self initArtTransfrom];
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
            
            void *imageRawData = [self getPixelsFromImage:_originImage.CGImage];
            
            _artImage = [self runArtTransfromWithHeight:height width:width input:imageRawData];
            
            free(imageRawData);
        }
        
        [_imageView setImage:_artImage];
        
        _isArt = YES;
    }
}

- (UIImage*)runArtTransfromWithHeight:(int)height width:(int)width input:(void*)uint8Input {
    NSAssert(height > 16 && width > 16, @"the input height and width is error!");
    
    [self configFloatBufferWithHeight:height width:width device:device];
    
    memcpy(buffer1.buffer.contents, uint8Input, 4 * height * width);
    
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    
    TensorShape inputShape;
    TensorShape outputShape;
    
    inputShape.dim0 = height;
    inputShape.dim1 = width;
    inputShape.dim2 = 4;
    
    outputShape.dim0 = height;
    outputShape.dim1 = width;
    outputShape.dim2 = 4;
    
    [rgbConvert1 computeWithCommandBuffer:commandBuffer
                                    input:buffer1.buffer
                               inputShape:inputShape
                                   output:buffer2.buffer
                              outputShape:outputShape];
    
    outputShape.dim2 = 32;
    
    [conv1 computeWithCommandBuffer:commandBuffer
                              input:buffer2.buffer
                         inputShape:inputShape
                             output:buffer1.buffer
                        outputShape:outputShape];
    
    inputShape = outputShape;
    
    [batchNorm1 computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [relu1      computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    
    outputShape.dim0 = ceil(1.0 * outputShape.dim0 / 2.0);
    outputShape.dim1 = ceil(1.0 * outputShape.dim1 / 2.0);
    outputShape.dim2 = 64;
    
    [conv2 computeWithCommandBuffer:commandBuffer
                              input:buffer1.buffer
                         inputShape:inputShape
                             output:buffer2.buffer
                        outputShape:outputShape];
    
    inputShape = outputShape;
    
    [batchNorm2 computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    [relu2      computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    
    outputShape.dim0 = ceil(1.0 * outputShape.dim0 / 2.0);
    outputShape.dim1 = ceil(1.0 * outputShape.dim1 / 2.0);
    outputShape.dim2 = 128;
    
    [conv3 computeWithCommandBuffer:commandBuffer
                              input:buffer2.buffer
                         inputShape:inputShape
                             output:buffer1.buffer
                        outputShape:outputShape];
    
    inputShape = outputShape;
    
    [batchNorm3 computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [relu3      computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    
    [res1 computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [res2 computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    [res3 computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [res4 computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    [res5 computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    
    outputShape.dim0 = outputShape.dim0 * 2;
    outputShape.dim1 = outputShape.dim1 * 2;
    outputShape.dim2 = 64;
    
    [transposeConv1 computeWithCommandBuffer:commandBuffer
                                       input:buffer2.buffer
                                  inputShape:inputShape
                                      output:buffer1.buffer
                                 outputShape:outputShape];
    
    inputShape = outputShape;
    
    [batchNorm4 computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [relu4      computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    
    outputShape.dim0 = outputShape.dim0 * 2;
    outputShape.dim1 = outputShape.dim1 * 2;
    outputShape.dim2 = 32;
    
    [transposeConv2 computeWithCommandBuffer:commandBuffer
                                       input:buffer1.buffer
                                  inputShape:inputShape
                                      output:buffer2.buffer
                                 outputShape:outputShape];
    
    inputShape = outputShape;
    
    [batchNorm5 computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    [relu5      computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    
    outputShape.dim2 = 4;
    
    [conv4 computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    
    inputShape = outputShape;
    
    [batchNorm6     computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [tanh1          computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    [linear1        computeWithCommandBuffer:commandBuffer input:buffer1.buffer inputShape:inputShape output:buffer2.buffer outputShape:outputShape];
    [rgbConvert2    computeWithCommandBuffer:commandBuffer input:buffer2.buffer inputShape:inputShape output:buffer1.buffer outputShape:outputShape];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    void *imageRawData = malloc(4 * outputShape.dim0 * outputShape.dim1);
    
    memcpy(imageRawData, buffer1.buffer.contents, 4 * outputShape.dim0 * outputShape.dim1);
    
    UIImage *artImage = [self getUIImageFromPixels:imageRawData width:outputShape.dim1 height:outputShape.dim0];
    
    return artImage;
}

- (void)configFloatBufferWithHeight:(int)height width:(int)width device:(id<MTLDevice>)device {
    [buffer1 configConvolutionMMWithHalfHeight:height width:width channel:3];
    [buffer2 configConvolutionMMWithHalfHeight:height width:width channel:3];
    
    [buffer1 configConvolutionMMWithHalfHeight:height width:width channel:32];
    [buffer2 configConvolutionMMWithHalfHeight:height width:width channel:32];
    
    height = ceil(1.0 * height / 2.0);
    width  = ceil(1.0 * width  / 2.0);
    
    [buffer1 configConvolutionMMWithHalfHeight:height width:width channel:64];
    [buffer2 configConvolutionMMWithHalfHeight:height width:width channel:64];
    
    height = ceil(1.0 * height / 2.0);
    width  = ceil(1.0 * width  / 2.0);
    
    [buffer1 configConvolutionMMWithHalfHeight:height width:width channel:128];
    [buffer2 configConvolutionMMWithHalfHeight:height width:width channel:128];
    
    height = 2 * height;
    width  = 2 * width;
    
    [buffer1 configConvolutionMMWithHalfHeight:height width:width channel:64];
    [buffer2 configConvolutionMMWithHalfHeight:height width:width channel:64];
    
    height = 2 * height;
    width  = 2 * width;
    
    [buffer1 configConvolutionMMWithHalfHeight:height width:width channel:32];
    [buffer2 configConvolutionMMWithHalfHeight:height width:width channel:32];
    
    [buffer1 configWithDevice:device];
    [buffer2 configWithDevice:device];
}

- (void)initArtTransfrom {
    /**load the LeNet model to test the Brouhaha*/
    NSBundle *mainBundle = [NSBundle mainBundle];
    
    device = MTLCreateSystemDefaultDevice();
    queue = [device newCommandQueue];
    
    NSError *libraryError = NULL;
    NSString *libraryFile = [mainBundle pathForResource:@"BrouhahaMetal" ofType:@"metallib"];
    library = [device newLibraryWithFile:libraryFile error:&libraryError];
    
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
    
    rgbConvert1 = [[BrouConvertFromuchar2halfLayer alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    conv1  = [[BrouConvolutionLayer_half alloc] initWithDevice:device
                                                        library:library
                                                    floatKernel:conv1_weight
                                                      floatBias:nil
                                                   inputChannel:3
                                                  outputChannel:32
                                                   kernelHeight:9
                                                    kernelWidth:9
                                                         padTop:4
                                                        padLeft:4
                                                        strideY:1
                                                        strideX:1];
    
    batchNorm1 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                   library:library
                                                                   epsilon:epsilon
                                                                floatAlpha:conv1_alpha
                                                                 floatBeta:conv1_beta
                                                                   channel:32];
    
    relu1 = [[BrouReLuLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    conv2 = [[BrouConvolutionMMLayer_half alloc] initWithDevice:device
                                                         library:library
                                                     floatKernel:conv2_weight
                                                       floatBias:nil
                                                    inputChannel:32
                                                   outputChannel:64
                                                    kernelHeight:3
                                                     kernelWidth:3
                                                          padTop:0
                                                         padLeft:0
                                                         strideY:2
                                                         strideX:2];
    
    batchNorm2 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                   library:library
                                                                   epsilon:epsilon
                                                                floatAlpha:conv2_alpha
                                                                 floatBeta:conv2_beta
                                                                   channel:64];
    
    relu2 = [[BrouReLuLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    conv3 = [[BrouConvolutionMMLayer_half alloc] initWithDevice:device
                                                         library:library
                                                     floatKernel:conv3_weight
                                                       floatBias:nil
                                                    inputChannel:64
                                                   outputChannel:128
                                                    kernelHeight:3
                                                     kernelWidth:3
                                                          padTop:0
                                                         padLeft:0
                                                         strideY:2
                                                         strideX:2];
    
    batchNorm3 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                   library:library
                                                                   epsilon:epsilon
                                                                floatAlpha:conv3_alpha
                                                                 floatBeta:conv3_beta
                                                                   channel:128];
    
    relu3 = [[BrouReLuLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    res1 = [[BrouResidualLayer_half alloc] initWithDevice:device
                                                   library:library
                                              floatWeight1:res1_conv1_weight
                                              floatWeight2:res1_conv2_weight
                                               floatAlpha1:res1_conv1_alpha
                                                floatBeta1:res1_conv1_beta
                                               floatAlpha2:res1_conv2_alpha
                                                floatBeta2:res1_conv2_beta
                                                   channel:128];
    
    res2 = [[BrouResidualLayer_half alloc] initWithDevice:device
                                                   library:library
                                              floatWeight1:res2_conv1_weight
                                              floatWeight2:res2_conv2_weight
                                               floatAlpha1:res2_conv1_alpha
                                                floatBeta1:res2_conv1_beta
                                               floatAlpha2:res2_conv2_alpha
                                                floatBeta2:res2_conv2_beta
                                                   channel:128];
    
    res3 = [[BrouResidualLayer_half alloc] initWithDevice:device
                                                   library:library
                                              floatWeight1:res3_conv1_weight
                                              floatWeight2:res3_conv2_weight
                                               floatAlpha1:res3_conv1_alpha
                                                floatBeta1:res3_conv1_beta
                                               floatAlpha2:res3_conv2_alpha
                                                floatBeta2:res3_conv2_beta
                                                   channel:128];
    
    res4 = [[BrouResidualLayer_half alloc] initWithDevice:device
                                                   library:library
                                              floatWeight1:res4_conv1_weight
                                              floatWeight2:res4_conv2_weight
                                               floatAlpha1:res4_conv1_alpha
                                                floatBeta1:res4_conv1_beta
                                               floatAlpha2:res4_conv2_alpha
                                                floatBeta2:res4_conv2_beta
                                                   channel:128];
    
    res5 = [[BrouResidualLayer_half alloc] initWithDevice:device
                                                   library:library
                                              floatWeight1:res5_conv1_weight
                                              floatWeight2:res5_conv2_weight
                                               floatAlpha1:res5_conv1_alpha
                                                floatBeta1:res5_conv1_beta
                                               floatAlpha2:res5_conv2_alpha
                                                floatBeta2:res5_conv2_beta
                                                   channel:128];
    
    transposeConv1 = [[BrouTransposedConvolutionMMLayer_half alloc] initWithDevice:device
                                                                            library:library
                                                                        floatKernel:transpose_conv1_weight
                                                                          floatBias:nil
                                                                 originInputChannel:64
                                                                originOutputChannel:128
                                                                 originKernelHeight:3
                                                                  originKernelWidth:3
                                                                       originPadTop:0
                                                                      originPadLeft:0
                                                                      originStrideY:2
                                                                      originStrideX:2];
    
    batchNorm4 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                   library:library
                                                                   epsilon:epsilon
                                                                floatAlpha:transpose_conv1_alpha
                                                                 floatBeta:transpose_conv1_beta
                                                                   channel:64];
    
    relu4 = [[BrouReLuLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    transposeConv2 = [[BrouTransposedConvolutionMMLayer_half alloc] initWithDevice:device
                                                                            library:library
                                                                        floatKernel:transpose_conv2_weight
                                                                          floatBias:nil
                                                                 originInputChannel:32
                                                                originOutputChannel:64
                                                                 originKernelHeight:3
                                                                  originKernelWidth:3
                                                                       originPadTop:0
                                                                      originPadLeft:0
                                                                      originStrideY:2
                                                                      originStrideX:2];
    
    batchNorm5 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                   library:library
                                                                   epsilon:epsilon
                                                                floatAlpha:transpose_conv2_alpha
                                                                 floatBeta:transpose_conv2_beta
                                                                   channel:32];
    
    relu5 = [[BrouReLuLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    conv4 = [[BrouConvolutionLayer_half alloc] initWithDevice:device
                                                       library:library
                                                   floatKernel:conv4_weight
                                                     floatBias:nil
                                                  inputChannel:32
                                                 outputChannel:3
                                                  kernelHeight:9
                                                   kernelWidth:9
                                                        padTop:4
                                                       padLeft:4
                                                       strideY:1
                                                       strideX:1];
    
    batchNorm6 = [[BrouBatchNormalizationLayer_half alloc] initWithDevice:device
                                                                   library:library
                                                                   epsilon:epsilon
                                                                floatAlpha:conv4_alpha
                                                                 floatBeta:conv4_beta
                                                                   channel:3];
    
    tanh1 = [[BrouTanHLayer_half alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    linear1 = [[BrouLinearLayer_half alloc] initWithDevice:device library:library a:150 b:255.0 / 2.0 dimensionType:Dimension3D];
    
    rgbConvert2 = [[BrouConvertFromhalf2ucharLayer alloc] initWithDevice:device library:library dimensionType:Dimension3D];
    
    munmap(conv1_weight, 32*9*9*3*4);
    munmap(conv1_alpha, 32*4);
    munmap(conv1_beta, 32*4);
    
    munmap(conv2_weight, 64*3*3*32*4);
    munmap(conv2_alpha, 64*4);
    munmap(conv2_beta, 64*4);
    
    munmap(conv3_weight, 128*3*3*64*4);
    munmap(conv3_alpha, 128*4);
    munmap(conv3_beta, 128*4);
    
    munmap(res1_conv1_weight, 128*3*3*128*4);
    munmap(res1_conv1_alpha, 128*4);
    munmap(res1_conv1_beta, 128*4);
    munmap(res1_conv2_weight, 128*3*3*128*4);
    munmap(res1_conv2_alpha, 128*4);
    munmap(res1_conv2_beta, 128*4);
    
    munmap(res2_conv1_weight, 128*3*3*128*4);
    munmap(res2_conv1_alpha, 128*4);
    munmap(res2_conv1_beta, 128*4);
    munmap(res2_conv2_weight, 128*3*3*128*4);
    munmap(res2_conv2_alpha, 128*4);
    munmap(res2_conv2_beta, 128*4);
    
    munmap(res3_conv1_weight, 128*3*3*128*4);
    munmap(res3_conv1_alpha, 128*4);
    munmap(res3_conv1_beta, 128*4);
    munmap(res3_conv2_weight, 128*3*3*128*4);
    munmap(res3_conv2_alpha, 128*4);
    munmap(res3_conv2_beta, 128*4);
    
    munmap(res4_conv1_weight, 128*3*3*128*4);
    munmap(res4_conv1_alpha, 128*4);
    munmap(res4_conv1_beta, 128*4);
    munmap(res4_conv2_weight, 128*3*3*128*4);
    munmap(res4_conv2_alpha, 128*4);
    munmap(res4_conv2_beta, 128*4);
    
    munmap(res5_conv1_weight, 128*3*3*128*4);
    munmap(res5_conv1_alpha, 128*4);
    munmap(res5_conv1_beta, 128*4);
    munmap(res5_conv2_weight, 128*3*3*128*4);
    munmap(res5_conv2_alpha, 128*4);
    munmap(res5_conv2_beta, 128*4);
    
    munmap(transpose_conv1_weight, 128*64*3*3*4);
    munmap(transpose_conv1_alpha, 64*4);
    munmap(transpose_conv1_beta, 64*4);
    
    munmap(transpose_conv2_weight, 64*32*3*3*4);
    munmap(transpose_conv2_alpha, 32*4);
    munmap(transpose_conv2_beta, 32*4);
    
    munmap(conv4_weight, 3*9*9*32*4);
    munmap(conv4_alpha, 3*4);
    munmap(conv4_beta, 3*4);
    
    buffer1 = [[BrouTemporaryBuffer alloc] init];
    buffer2 = [[BrouTemporaryBuffer alloc] init];
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

@end
