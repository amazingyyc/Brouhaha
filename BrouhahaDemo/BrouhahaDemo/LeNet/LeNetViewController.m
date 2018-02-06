#import "sys/mman.h"
#import "Brouhaha.h"
#import "LeNetViewController.h"
#import "PaintView.h"

@interface LeNetViewController () {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;
    id<MTLLibrary> _library;
    
    BrouConvolutionMMLayer_float *_conv0;
    BrouMaxPoolingLayer_float *_maxPooling0;
    BrouAddBiasLayer_float *_add0;
    BrouTanHLayer_float *_tanh0;
    
    BrouConvolutionMMLayer_float *_conv1;
    BrouMaxPoolingLayer_float *_maxPooling1;
    BrouAddBiasLayer_float *_add1;
    BrouTanHLayer_float *_tanh1;
    
    BrouConvolutionMMLayer_float *_fullConnect0;
    BrouTanHLayer_float *_tanh2;
    BrouFullConnectLayer_float *_fullConnect1;
    
    BrouShareBuffer *_temporaryBuffer;
    
    id<BrouTensor> _conv0Input;
    id<BrouTensor> _conv0Output;
    id<BrouTensor> _maxPooling0Output;
    id<BrouTensor> _add0Output;
    id<BrouTensor> _tanh0Output;
    id<BrouTensor> _conv1Output;
    id<BrouTensor> _maxPooling1Output;
    id<BrouTensor> _add1Output;
    id<BrouTensor> _tanh1Output;
    id<BrouTensor> _fullConnect0Output;
    id<BrouTensor> _tanh2input;
    id<BrouTensor> _tanh2Output;
    id<BrouTensor> _fullConnect1Output;

    /**a temp buffer*/
    // id<MTLBuffer> _buffer0;
    // id<MTLBuffer> _buffer1;
}

@property(nonatomic, strong) PaintView *paintView;
@property(nonatomic, strong) UILabel *label;
@property(nonatomic, strong) UILabel *result;
@property(nonatomic, strong) UIButton *closeButton;
@property(nonatomic, strong) UIButton *clearButton;
@property(nonatomic, strong) UIButton *recognizeButton;

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
    
    [self initLeNet];
    
    /**test*/
    /**load the LeNet model to test the Brouhaha*/
//    NSBundle *mainBundle = [NSBundle mainBundle];
//    float *testX = [self readBinaryFile:[mainBundle pathForResource:@"testX100" ofType:@""] length:100*28*28*4];
//
//    id<MTLCommandBuffer> commandBuffer = [_queue commandBuffer];
//    [self runLeNetWithInput:testX commandBuffer:commandBuffer];
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

- (void*)getRawDataFromPaintView {
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    void *uint8Data = malloc(28 * 28);
    
    CGContextRef context = CGBitmapContextCreate(uint8Data,
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
    
    float *float32RawData = malloc(28 * 28 * sizeof(float));
    
    for (int i = 0; i < 28 * 28; ++i) {
        float float32 = ((uint8_t*)uint8Data)[i];
        
        float32RawData[i] = 1.0 - float32 / 255.0;
    }
    
    free(uint8Data);
    
    return float32RawData;
}

- (void)recognizeDigitNumber {
    void *rawData = [self getRawDataFromPaintView];
    
    UInt64 t1 = [self getCurrentTimeNow];
    
    id<MTLCommandBuffer> commandBuffer = [_queue commandBuffer];
    int digitNumber = [self runLeNetWithInput:rawData commandBuffer:commandBuffer];
    
    UInt64 t2 = [self getCurrentTimeNow];
    
    free(rawData);
    
    NSString *text = [NSString stringWithFormat:@"the digit number is:%d, cost time:%llu ms", digitNumber, (t2  -t1)];
    
    _result.text = text;
    [_result sizeToFit];
}

- (UInt64)getCurrentTimeNow {
    return [[NSDate date] timeIntervalSince1970] * 1000;
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
}

- (void)initLeNet {
    /**load the LeNet model to test the Brouhaha*/
    NSBundle *mainBundle = [NSBundle mainBundle];
    
    _device = MTLCreateSystemDefaultDevice();
    _queue = [_device newCommandQueue];
    
    NSError *libraryError = NULL;
    NSString *libraryFile = [mainBundle pathForResource:@"BrouhahaMetal" ofType:@"metallib"];
    _library = [_device newLibraryWithFile:libraryFile error:&libraryError];
    
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
    
    /**
     * input (28, 28, 1) (3136)
     */
    _conv0 = [[BrouConvolutionMMLayer_float alloc] initWithDevice:_device
                                                        library:_library
                                                    floatKernel:w0
                                                      floatBias:NULL
                                                   inputChannel:1
                                                  outputChannel:20
                                                   kernelHeight:5
                                                    kernelWidth:5
                                                         padTop:0
                                                        padLeft:0
                                                        strideY:1
                                                        strideX:1];
    
    /**
     * input (24, 24, 20) (11520)
     */
    _maxPooling0 = [[BrouMaxPoolingLayer_float alloc] initWithDevice:_device
                                                             library:_library
                                                        kernelHeight:2
                                                         kernelWidth:2
                                                              padTop:0
                                                             padLeft:0
                                                             strideY:2
                                                             strideX:2];
    
    /**
     * input (12, 12, 20)
     */
    _add0 = [[BrouAddBiasLayer_float alloc] initWithDevice:_device
                                                   library:_library
                                                 floatBias:b0
                                                biasLength:20
                                             dimensionType:Dimension3D];
    
    _tanh0 = [[BrouTanHLayer_float alloc] initWithDevice:_device library:_library dimensionType:Dimension3D];
    
    /**
     * input (12, 12, 20)
     */
    _conv1 = [[BrouConvolutionMMLayer_float alloc] initWithDevice:_device
                                                        library:_library
                                                    floatKernel:w1
                                                      floatBias:NULL
                                                   inputChannel:20
                                                  outputChannel:50
                                                   kernelHeight:5
                                                    kernelWidth:5
                                                         padTop:0
                                                        padLeft:0
                                                        strideY:1
                                                        strideX:1];
    /**
     * input (8, 8, 50) (3200)
     */
    _maxPooling1 = [[BrouMaxPoolingLayer_float alloc] initWithDevice:_device
                                                             library:_library
                                                        kernelHeight:2
                                                         kernelWidth:2
                                                              padTop:0
                                                             padLeft:0
                                                             strideY:2
                                                             strideX:2];
    
    /**
     * input (4, 4, 50)
     */
    _add1 = [[BrouAddBiasLayer_float alloc] initWithDevice:_device
                                                   library:_library
                                                 floatBias:b1
                                                biasLength:50
                                             dimensionType:Dimension3D];
    
    _tanh1 = [[BrouTanHLayer_float alloc] initWithDevice:_device library:_library dimensionType:Dimension3D];
    
    /**
     * input (4, 4, 50) (4 * 4 * 500 = 8000)
     */
    _fullConnect0 = [[BrouConvolutionMMLayer_float alloc] initWithDevice:_device
                                                               library:_library
                                                           floatKernel:w2
                                                             floatBias:b2
                                                          inputChannel:50
                                                         outputChannel:500
                                                          kernelHeight:4
                                                           kernelWidth:4
                                                                padTop:0
                                                               padLeft:0
                                                               strideY:1
                                                               strideX:1];
    
    _tanh2 = [[BrouTanHLayer_float alloc] initWithDevice:_device library:_library dimensionType:Dimension1D];
    
    /**
     * input (500, 1)
     */
    _fullConnect1 = [[BrouFullConnectLayer_float alloc] initWithDevice:_device
                                                               library:_library
                                                          floatWeights:w3
                                                             floatBias:b3
                                                         intputChannel:500
                                                         outputChannel:10];

    munmap(w0, 20*1*5*5*4);
    munmap(b0, 20*4);
    munmap(w1, 50*20*5*5*4);
    munmap(b1, 50*4);
    munmap(w2, 500*800*4);
    munmap(b2, 500*4);
    munmap(w3, 10*500*4);
    munmap(b3, 10*4);
    
    _temporaryBuffer = [BrouShareBuffer defaultWithDevice:_device];
    
    _conv0Input         = [BrouTemporaryTensor_float initWithHeight:28 width:28 channel:1 temporaryBufer:_temporaryBuffer];
    _conv0Output        = [BrouTemporaryTensor_float initWithHeight:24 width:24 channel:20 temporaryBufer:_temporaryBuffer];
    _maxPooling0Output  = [BrouTemporaryTensor_float initWithHeight:12 width:12 channel:20 temporaryBufer:_temporaryBuffer];
    _add0Output         = [BrouTemporaryTensor_float initWithHeight:12 width:12 channel:20 temporaryBufer:_temporaryBuffer];
    _tanh0Output        = [BrouTemporaryTensor_float initWithHeight:12 width:12 channel:20 temporaryBufer:_temporaryBuffer];
    _conv1Output        = [BrouTemporaryTensor_float initWithHeight:8 width:8 channel:50 temporaryBufer:_temporaryBuffer];
    _maxPooling1Output  = [BrouTemporaryTensor_float initWithHeight:4 width:4 channel:50 temporaryBufer:_temporaryBuffer];
    _add1Output         = [BrouTemporaryTensor_float initWithHeight:4 width:4 channel:50 temporaryBufer:_temporaryBuffer];
    _tanh1Output        = [BrouTemporaryTensor_float initWithHeight:4 width:4 channel:50 temporaryBufer:_temporaryBuffer];
    _fullConnect0Output = [BrouTemporaryTensor_float initWithHeight:1 width:1 channel:500 temporaryBufer:_temporaryBuffer];
    
    _tanh2input         = [BrouTemporaryTensor_float initWithLength:500 anotherTemporaryTensor:_fullConnect0Output];
    
    _tanh2Output        = [BrouTemporaryTensor_float initWithLength:500 temporaryBufer:_temporaryBuffer];
    _fullConnect1Output = [BrouTemporaryTensor_float initWithLength:10 temporaryBufer:_temporaryBuffer];
}

/**
 * the input is (28, 28, 1) float gray image
 */
- (int)runLeNetWithInput:(float*)input commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    float *floatInput = _conv0Input.tensorBuffer.contents;
    
    for (int i = 0; i < 28 * 28; ++i) {
        floatInput[4 * i] = input[i];
    }
    
    [_conv0        computeCommandBuffer:commandBuffer input:_conv0Input         output:_conv0Output];
    [_maxPooling0  computeCommandBuffer:commandBuffer input:_conv0Output        output:_maxPooling0Output];
    [_add0         computeCommandBuffer:commandBuffer input:_maxPooling0Output  output:_add0Output];
    [_tanh0        computeCommandBuffer:commandBuffer input:_add0Output         output:_tanh0Output];
    [_conv1        computeCommandBuffer:commandBuffer input:_tanh0Output        output:_conv1Output];
    [_maxPooling1  computeCommandBuffer:commandBuffer input:_conv1Output        output:_maxPooling1Output];
    [_add1         computeCommandBuffer:commandBuffer input:_maxPooling1Output  output:_add1Output];
    [_tanh1        computeCommandBuffer:commandBuffer input:_add1Output         output:_tanh1Output];
    [_fullConnect0 computeCommandBuffer:commandBuffer input:_tanh1Output        output:_fullConnect0Output];
    [_tanh2        computeCommandBuffer:commandBuffer input:_tanh2input         output:_tanh2Output];
    [_fullConnect1 computeCommandBuffer:commandBuffer input:_tanh2Output        output:_fullConnect1Output];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    float *floatOutput = _fullConnect1Output.tensorBuffer.contents;
    int max = 0;
    for (int i = 0; i < 10; ++i) {
        if (floatOutput[i] > floatOutput[max]) {
            max = i;
        }
    }

    return max;
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








