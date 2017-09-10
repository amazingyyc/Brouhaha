#import "BrouTemporaryBuffer.h"

@implementation BrouTemporaryBuffer

- (instancetype)init {
    self = [super init];
    
    if (self) {
        _length = 0;
    }
    
    return self;
}

- (void)configWithFloatLength:(int)length {
    NSAssert(length > 0, @"length must > 0");
    
    int lengthX4 = (length + 3) / 4 * 4;
    
    [self configWithLength:sizeof(float) * lengthX4];
}

- (void)configWithFloatHeight:(int)height width:(int)width {
    NSAssert(height > 0, @"height must > 0");
    NSAssert(width > 0, @"width must > 0");
    
    int widthX4 = (width + 3) / 4 * 4;
    
    [self configWithLength:sizeof(float) * height * widthX4];
}

- (void)configWithFloatHeight:(int)height width:(int)width channel:(int)channel {
    NSAssert(height > 0, @"height must > 0");
    NSAssert(width > 0, @"width must > 0");
    NSAssert(channel > 0, @"channel must > 0");
    
    int channelX4 = (channel + 3) / 4 * 4;
    
    [self configWithLength:sizeof(float) * height * width * channelX4];
}

- (void)configWithHalfLength:(int)length {
    NSAssert(length > 0, @"length must > 0");
    
    int lengthX4 = (length + 3) / 4 * 4;
    
    [self configWithLength:sizeof(uint16_t) * lengthX4];
}

- (void)configWithHalfHeight:(int)height width:(int)width {
    NSAssert(height > 0, @"height must > 0");
    NSAssert(width > 0, @"width must > 0");
    
    int widthX4 = (width + 3) / 4 * 4;
    
    [self configWithLength:sizeof(uint16_t) * height * widthX4];
}

- (void)configWithHalfHeight:(int)height width:(int)width channel:(int)channel {
    NSAssert(height > 0, @"height must > 0");
    NSAssert(width > 0, @"width must > 0");
    NSAssert(channel > 0, @"channel must > 0");
    
    int channelX4 = (channel + 3) / 4 * 4;

    [self configWithLength:sizeof(uint16_t) * height * width * channelX4];
}

- (void)configConvolutionMMWithFloatHeight:(int)height width:(int)width channel:(int)channel {
    NSAssert(height > 0, @"height must > 0");
    NSAssert(width > 0, @"width must > 0");
    NSAssert(channel > 0, @"channel must > 0");
    
    int heightXwidthX4 = (height * width + 3) / 4 * 4;
    int channelX4 = (channel + 3) / 4 * 4;
    
    [self configWithLength:sizeof(float) * heightXwidthX4 * channelX4];
}

- (void)configConvolutionMMWithHalfHeight:(int)height width:(int)width channel:(int)channel {
    NSAssert(height > 0, @"height must > 0");
    NSAssert(width > 0, @"width must > 0");
    NSAssert(channel > 0, @"channel must > 0");
    
    int heightXwidthX4 = (height * width + 3) / 4 * 4;
    int channelX4 = (channel + 3) / 4 * 4;
    
    [self configWithLength:sizeof(uint16_t) * heightXwidthX4 * channelX4];
}

- (void)configWithLength:(int)length {
    NSAssert(length > 0, @"length must > 0");
    
    if (_length < length) {
        _length = length;
    }
}

- (void)configWithDevice:(id<MTLDevice>)device {
    if (!_buffer || _buffer.length < _length) {
        _buffer = [device newBufferWithLength:_length
                                      options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
    }
}

@end
