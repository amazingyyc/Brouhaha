#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

@interface BROU_OBJECT(FullConnectLayer) : BrouLayer

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  floatWeights:(void*)floatWeight
                     floatBias:(void*)floatBias
                 intputChannel:(int)inputChannel
                 outputChannel:(int)outputChannel;

@end

#endif











