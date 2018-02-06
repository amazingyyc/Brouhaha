#import "sys/mman.h"
#import "ArtTransformHalfViewController.h"
#import "Brouhaha.h"
#import "BrouResidualLayer_half.h"
#import "BrouShareBuffer.h"

@interface ArtTransformHalfViewController () {
    BrouConvertFromuchar2halfLayer *rgbConvert1;
    
    BrouConvolutionLayer_half *conv1;
    BrouBatchNormalizationLayer_half *batchNorm1;
    BrouReLuLayer_half *relu1;
    
    BrouConvolutionLayer_half *conv2;
    BrouBatchNormalizationLayer_half *batchNorm2;
    BrouReLuLayer_half *relu2;
    BrouConvolutionLayer_half *conv3;
    BrouBatchNormalizationLayer_half *batchNorm3;
    BrouReLuLayer_half *relu3;
    
    BrouResidualLayer_half *res1;
    BrouResidualLayer_half *res2;
    BrouResidualLayer_half *res3;
    BrouResidualLayer_half *res4;
    BrouResidualLayer_half *res5;
    
    BrouTransposedConvolutionLayer_half *transposeConv1;
    
    BrouBatchNormalizationLayer_half *batchNorm4;
    BrouReLuLayer_half *relu4;
    
    BrouTransposedConvolutionLayer_half *transposeConv2;
    
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
    
    id<BrouTensor> rgbConvert1Input;
    id<BrouTensor> rgbConvert1Output;
    
    id<BrouTensor> conv1Output;
    id<BrouTensor> batchNorm1Output;
    id<BrouTensor> relu1Output;
    
    id<BrouTensor> conv2Output;
    id<BrouTensor> batchNorm2Output;
    id<BrouTensor> relu2Output;
    
    id<BrouTensor> conv3Output;
    id<BrouTensor> batchNorm3Output;
    id<BrouTensor> relu3Output;
    
    id<BrouTensor> res1Output;
    id<BrouTensor> res2Output;
    id<BrouTensor> res3Output;
    id<BrouTensor> res4Output;
    id<BrouTensor> res5Output;
    
    id<BrouTensor> transposeConv1Output;
    
    id<BrouTensor> batchNorm4Output;
    id<BrouTensor> relu4Output;
    
    id<BrouTensor> transposeConv2Output;
    
    id<BrouTensor> batchNorm5Output;
    id<BrouTensor> relu5Output;
    
    id<BrouTensor> conv4Output;
    
    id<BrouTensor> batchNorm6Output;
    
    id<BrouTensor> tanh1Output;
    id<BrouTensor> linear1Output;
    id<BrouTensor> rgbConvert2Output;
    
    BrouShareBuffer *shareBuffer;
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

- (void)configFloatBufferWithHeight:(int)height width:(int)width device:(id<MTLDevice>)device {
    shareBuffer = [BrouShareBuffer defaultWithDevice:device];
    
    rgbConvert1Input = [BrouTemporaryTensor_half initWithHeight:height width:width channel:4 temporaryBufer:shareBuffer];
    
    rgbConvert1Output = [BrouTemporaryTensor_half initWithHeight:height width:width channel:4 temporaryBufer:shareBuffer];
    
    conv1Output      = [BrouTemporaryTensor_half initWithHeight:height width:width channel:32 temporaryBufer:shareBuffer];
    batchNorm1Output = [BrouTemporaryTensor_half initWithHeight:height width:width channel:32 temporaryBufer:shareBuffer];
    relu1Output      = [BrouTemporaryTensor_half initWithHeight:height width:width channel:32 temporaryBufer:shareBuffer];
    
    int dim0 = ceil(1.0 * height / 2.0);
    int dim1 = ceil(1.0 * width  / 2.0);
    int dim2 = 64;
    
    conv2Output      = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    batchNorm2Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    relu2Output      = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    dim0 = ceil(1.0 * dim0 / 2.0);
    dim1 = ceil(1.0 * dim1 / 2.0);
    dim2 = 128;
    
    conv3Output      = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    batchNorm3Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    relu3Output      = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    res1Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    res2Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    res3Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    res4Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    res5Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    dim0 = dim0 * 2;
    dim1 = dim1 * 2;
    dim2 = 64;
    
    transposeConv1Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    batchNorm4Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    relu4Output      = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    dim0 = dim0 * 2;
    dim1 = dim1 * 2;
    dim2 = 32;
    
    transposeConv2Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    batchNorm5Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    relu5Output      = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    dim2 = 4;
    
    conv4Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    batchNorm6Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    
    tanh1Output       = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    linear1Output     = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
    rgbConvert2Output = [BrouTemporaryTensor_half initWithHeight:dim0 width:dim1 channel:dim2 temporaryBufer:shareBuffer];
}

- (UIImage*)runArtTransfromWithHeight:(int)height width:(int)width input:(void*)uint8Input {
    NSAssert(height > 16 && width > 16, @"the input height and width is error!");
    
    [self configFloatBufferWithHeight:height width:width device:device];
    
    memcpy(rgbConvert1Input.tensorBuffer.contents, uint8Input, 4 * height * width);
    
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    
    [rgbConvert1 computeCommandBuffer:commandBuffer input:rgbConvert1Input output:rgbConvert1Output];
    [conv1       computeCommandBuffer:commandBuffer input:rgbConvert1Output output:conv1Output];
    [batchNorm1  computeCommandBuffer:commandBuffer input:conv1Output output:batchNorm1Output];
    [relu1       computeCommandBuffer:commandBuffer input:batchNorm1Output output:relu1Output];
    
    [conv2       computeCommandBuffer:commandBuffer input:relu1Output output:conv2Output];
    [batchNorm2  computeCommandBuffer:commandBuffer input:conv2Output output:batchNorm2Output];
    [relu2       computeCommandBuffer:commandBuffer input:batchNorm2Output output:relu2Output];
    
    [conv3       computeCommandBuffer:commandBuffer input:relu2Output output:conv3Output];
    [batchNorm3  computeCommandBuffer:commandBuffer input:conv3Output output:batchNorm3Output];
    [relu3       computeCommandBuffer:commandBuffer input:batchNorm3Output output:relu3Output];
    
    [res1 computeCommandBuffer:commandBuffer input:relu3Output output:res1Output];
    [res2 computeCommandBuffer:commandBuffer input:res1Output output:res2Output];
    [res3 computeCommandBuffer:commandBuffer input:res2Output output:res3Output];
    [res4 computeCommandBuffer:commandBuffer input:res3Output output:res4Output];
    [res5 computeCommandBuffer:commandBuffer input:res4Output output:res5Output];
    
    [transposeConv1 computeCommandBuffer:commandBuffer input:res5Output output:transposeConv1Output];
    
    [batchNorm4  computeCommandBuffer:commandBuffer input:transposeConv1Output output:batchNorm4Output];
    [relu4       computeCommandBuffer:commandBuffer input:batchNorm4Output output:relu4Output];
    
    [transposeConv2 computeCommandBuffer:commandBuffer input:relu4Output output:transposeConv2Output];
    [batchNorm5     computeCommandBuffer:commandBuffer input:transposeConv2Output output:batchNorm5Output];
    [relu5          computeCommandBuffer:commandBuffer input:batchNorm5Output output:relu5Output];
    
    [conv4       computeCommandBuffer:commandBuffer input:relu5Output output:conv4Output];
    [batchNorm6  computeCommandBuffer:commandBuffer input:conv4Output output:batchNorm6Output];
    [tanh1       computeCommandBuffer:commandBuffer input:batchNorm6Output output:tanh1Output];
    [linear1     computeCommandBuffer:commandBuffer input:tanh1Output output:linear1Output];
    [rgbConvert2 computeCommandBuffer:commandBuffer input:linear1Output output:rgbConvert2Output];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    

    void *imageRawData = malloc(4 * rgbConvert2Output.dim0 * rgbConvert2Output.dim1);
    
    memcpy(imageRawData, rgbConvert2Output.tensorBuffer.contents, 4 * rgbConvert2Output.dim0 * rgbConvert2Output.dim1);
    
    UIImage *artImage = [self getUIImageFromPixels:imageRawData width:rgbConvert2Output.dim1 height:rgbConvert2Output.dim0];
    
    return artImage;
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
    
    conv2 = [[BrouConvolutionLayer_half alloc] initWithDevice:device
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
    
    conv3 = [[BrouConvolutionLayer_half alloc] initWithDevice:device
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
    
    transposeConv1 = [[BrouTransposedConvolutionLayer_half alloc] initWithDevice:device
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
    
    transposeConv2 = [[BrouTransposedConvolutionLayer_half alloc] initWithDevice:device
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
                                                                   channel:4];
    
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
