#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(TemporaryTensor) : NSObject <BrouTensor>

/**
 the init function
 */
+ (instancetype)initWithLength:(int)length temporaryBufer:(BrouShareBuffer*)temporaryBuffer;
+ (instancetype)initWithHeight:(int)height width:(int)width temporaryBufer:(BrouShareBuffer*)temporaryBuffer;
+ (instancetype)initWithHeight:(int)height width:(int)width channel:(int)channel temporaryBufer:(BrouShareBuffer*)temporaryBuffer;

/**
 init a TemporaryTensor with another TemporaryTensor
 this Tensor and another TemporaryTensor will point to the same MTLBuffer
 and the bytesCount must be less than another TemporaryTensor's bytesCount
 */
+ (instancetype)initWithLength:(int)length
        anotherTemporaryTensor:(BROU_OBJECT(TemporaryTensor)*)anotherTensor;

+ (instancetype)initWithHeight:(int)height
                         width:(int)width
        anotherTemporaryTensor:(BROU_OBJECT(TemporaryTensor)*)anotherTensor;

+ (instancetype)initWithHeight:(int)height
                         width:(int)width
                       channel:(int)channel
        anotherTemporaryTensor:(BROU_OBJECT(TemporaryTensor)*)anotherTensor;

- (NSUInteger)bytesCount;
- (NSNumber*)temporaryBufferId;
- (BrouShareBuffer*)temporaryBuffer;

@end

#endif
