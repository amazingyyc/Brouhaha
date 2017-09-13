#if defined(type) && defined(real) && defined(BROU_METAL) && defined(BROU_OBJECT)

/**
 * in testing the mean and variance will be knowed
 * if don't the mean and variance will be calculate by
 * brouCalculateMeanAndVariance3D
 *
 * the alpha and beta is knowed
 *
 * output = alpha * (input - mean) / (sqrt(variance + epsilon)) + beta
 *
 * this layer is just fort the CNN batch normalization for now!!!
 */
@interface BROU_OBJECT(BatchNormalizationLayer) : BrouLayer {
    int _channel;
    int _channelX4;
    
    float _floatEpison;
    
    id<MTLBuffer> _bnShape;
    id<MTLBuffer> _shape;
    
    id<MTLBuffer> _mean;
    id<MTLBuffer> _variance;
    
    id<MTLBuffer> _alpha;
    id<MTLBuffer> _beta;
    
    NSString *_calculateMeanVarianceFunctionName;
    id<MTLComputePipelineState> _calculateMeanVariancePipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                       epsilon:(float)epsilon
                    floatAlpha:(void*)floatAlpha
                     floatBeta:(void*)floatBeta
                       channel:(int)channel;

@end

#endif
