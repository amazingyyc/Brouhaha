#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(UniqueTensor) : NSObject <BrouTensor>

/**
 the init function
 */
+ (instancetype)initWithLength:(int)length device:(id<MTLDevice>)device;
+ (instancetype)initWithHeight:(int)height width:(int)width device:(id<MTLDevice>)device;
+ (instancetype)initWithHeight:(int)height width:(int)width channel:(int)channel device:(id<MTLDevice>)device;

@end

#endif
