#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

/**
 the BrouTesnor a simple wrapper of memory
 */
@interface BROU_OBJECT(UniqueTensor)() {
    /**the MTL buffer, point to a memory*/
    id<MTLBuffer> _buffer;
    
    /**the dimension if the tensor*/
    NSUInteger _dimension;
    
    /**store the dim of the tensor's dimension*/
    NSArray<NSNumber*> *_dims;
    
    /**the length of the Tensor*/
    NSUInteger _length;
    
    /**
     for the preformance of Metal, the dimension's shape will be changed
     like a 1D Tensor and the dim[0] is 15
     The Tensor will malloc 16-memory to store the data
     so the length of this Tensor is 15, but the lengthX4 is 16
     
     When init the Tensor, innermost dimension(d) will be normalizated to dX4 the dX4 is not less than d and must be divided by 4 without remainder
     like a Tensor's diemension is (3, 5, 15)
     It will malloc a memeory is (2, 5, 16)
     so the length is 3 * 5 * 15
     but the lengthX4 is 3 * 5 * 16
     */
    NSUInteger _lengthX4;
    
    NSUInteger _bytesCount;
}

@end

@implementation BROU_OBJECT(UniqueTensor)

+ (instancetype)initWithLength:(int)length device:(id<MTLDevice>)device {
    NSMutableArray<NSNumber*> *dims = [[NSMutableArray<NSNumber*> alloc] init];
    [dims addObject:[NSNumber numberWithInt:length]];
    
    return [[BROU_OBJECT(UniqueTensor) alloc] initWithDimsArray:dims device:device];
}

+ (instancetype)initWithHeight:(int)height width:(int)width device:(id<MTLDevice>)device {
    NSMutableArray<NSNumber*> *dims = [[NSMutableArray<NSNumber*> alloc] init];
    [dims addObject:[NSNumber numberWithInt:height]];
    [dims addObject:[NSNumber numberWithInt:width]];

    return [[BROU_OBJECT(UniqueTensor) alloc] initWithDimsArray:dims device:device];
}

+ (instancetype)initWithHeight:(int)height width:(int)width channel:(int)channel device:(id<MTLDevice>)device {
    NSMutableArray<NSNumber*> *dims = [[NSMutableArray<NSNumber*> alloc] init];
    [dims addObject:[NSNumber numberWithInt:height]];
    [dims addObject:[NSNumber numberWithInt:width]];
    [dims addObject:[NSNumber numberWithInt:channel]];
    
    return [[BROU_OBJECT(UniqueTensor) alloc] initWithDimsArray:dims device:device];
}

- (instancetype)initWithDimsArray:(NSMutableArray<NSNumber*>*)dimArray device:(id<MTLDevice>)device {
    NSAssert(dimArray.count > 0, @"the dimension of Tensor must be > 0");
    
    self = [super init];
    
    if (self) {
        /**get the dim*/
        _dimension = dimArray.count;
        
        /**calcualte the length*/
        _length = 1;
        
        for (int i = 0; i < _dimension; ++i) {
            NSAssert(dimArray[i].intValue > 0, @"the dimension of Tensor must be > 0");

            _length *= dimArray[i].intValue;
        }
        
        int lastDim = [dimArray lastObject].intValue;
        int lastDimX4 = (lastDim + 3) / 4 * 4;
        
        _lengthX4 = _length / lastDim * lastDimX4;
        
        /**add to the last*/
        [dimArray addObject:[[NSNumber alloc] initWithInt:lastDimX4]];
        
        _dims = [[NSArray alloc] initWithArray:dimArray];
        
        _bytesCount = _lengthX4 * sizeof(type);
        
        /**malloc the memory*/
        if (@available(iOS 9.0, *)) {
            _buffer = [device newBufferWithLength:_bytesCount
                                          options:MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared];
        } else {
            _buffer = [device newBufferWithLength:_bytesCount
                                          options:MTLResourceCPUCacheModeDefaultCache];
        }
    }
    
    return self;
}

- (NSUInteger)dimension {
    return _dimension;
}

- (int)height {
    return [self dim0];
}

- (int)width {
    return [self dim1];
}

- (int)channel {
    return [self dim2];
}

- (int)dim0 {
    NSAssert(_dimension >= 1, @"the tensor has no dim0");
    
    return [_dims objectAtIndex:0].intValue;
}

- (int)dim1 {
    NSAssert(_dimension >= 2, @"the tensor has no dim1");
    
    return [_dims objectAtIndex:1].intValue;
}

- (int)dim2 {
    NSAssert(_dimension >= 3, @"the tensor has no dim2");
    
    return [_dims objectAtIndex:2].intValue;
}

- (int)dim:(int)dim {
    NSAssert(_dimension > dim, @"the dimension is error");
    
    return [_dims objectAtIndex:dim].intValue;
}

- (int)innermostDim {
    return [_dims objectAtIndex:_dimension - 1].intValue;
}

- (int)innermostDimX4 {
    return [_dims objectAtIndex:_dimension].intValue;
}

- (id<MTLBuffer>)tensorBuffer {
    return _buffer;
}

@end

#endif
