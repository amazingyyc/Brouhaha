#import "BrouShareBuffer.h"

/**
 BrouTemporaryBuffer will init a lot of MTLBuffer
 and some BrouTensor will point to a smae Buffer
 the step is:
 BrouTemporaryBuffer will init size-MTLBuffers.
 and when a BrouTemporaryTensor bind to a BrouTemporaryBuffer it will bind to one of the size-MTLBuffers
 so the two BrouTemporaryTensor may point to the same MTLBuffer
 */
@interface BrouShareBuffer() {
    id<MTLDevice> _device;
    
    /**_lifeTime means the temp MTLBuffer's life, default is 1*/
    NSUInteger _lifeTime;
    
    /**_size = _lifeTime + 1*/
    NSUInteger _size;
    
    /**the cur index of the buffer*/
    NSUInteger _curIndex;
    
    /**store the bytes count of MTLBuffer*/
    NSMutableArray<NSNumber*> *_bufferBytesCounts;
    
    /**store the real MTLBuffer*/
    NSMutableDictionary<NSNumber*, id<MTLBuffer>> *_temporaryBuffers;
}

@end

@implementation BrouShareBuffer

+ (instancetype)defaultWithDevice:(id<MTLDevice>)device {
    return [BrouShareBuffer initWithLifeTime:1 device:device];
}

+ (instancetype)initWithLifeTime:(NSUInteger)time device:(id<MTLDevice>)device {
    return [[BrouShareBuffer alloc] initWithLifeTime:time device:device];
}

- (instancetype)initWithLifeTime:(NSUInteger)lifeTime device:(id<MTLDevice>)device {
    NSAssert(lifeTime > 0, @"the life time must be > 0");
    
    self = [super init];
    
    if (self) {
        _device   = device;
        _lifeTime = lifeTime;
        _size     = _lifeTime + 1;
        _curIndex = 0;
        
        _temporaryBuffers  = [[NSMutableDictionary<NSNumber*, id<MTLBuffer>> alloc] init];
        _bufferBytesCounts = [[NSMutableArray<NSNumber*> alloc] init];
        
        for (NSUInteger i = 0; i < _size; ++i) {
            [_bufferBytesCounts addObject:[NSNumber numberWithUnsignedInteger:0]];
        }
    }
    
    return self;
}

- (NSNumber*)bindWithBytesCounts:(NSUInteger)bytesCount {
    if (_bufferBytesCounts[_curIndex].unsignedIntegerValue < bytesCount) {
        _bufferBytesCounts[_curIndex] = [NSNumber numberWithUnsignedInteger:bytesCount];
    }
    
    NSNumber *bindId = [NSNumber numberWithUnsignedInteger:_curIndex];
    
    _curIndex = (_curIndex + 1) % _size;
    
    return bindId;
}

- (id<MTLBuffer>)getBindBufferById:(NSNumber*)bindId {
    NSAssert(bindId.unsignedIntegerValue >= 0 && bindId.unsignedIntegerValue < _size, @"the bindId is error");
    
    if (nil == [_temporaryBuffers objectForKey:bindId]) {
        if (@available(iOS 9.0, *)) {
            id<MTLBuffer> buffer = [_device newBufferWithLength:_bufferBytesCounts[bindId.unsignedIntegerValue].unsignedIntegerValue
                                                        options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
            
            [_temporaryBuffers setObject:buffer forKey:bindId];
        } else {
            id<MTLBuffer> buffer = [_device newBufferWithLength:_bufferBytesCounts[bindId.unsignedIntegerValue].unsignedIntegerValue
                                                        options:MTLResourceCPUCacheModeDefaultCache];
            
            [_temporaryBuffers setObject:buffer forKey:bindId];
        }
    }
    
    return [_temporaryBuffers objectForKey:bindId];
}

@end









