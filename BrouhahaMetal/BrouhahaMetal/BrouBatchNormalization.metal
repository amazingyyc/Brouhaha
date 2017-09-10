#if defined(real) && defined(real4) && defined(BROU)

/**
 * in testing the mean and variance will be knowed
 * if don't know the training mean and variance, the mean and variance will be calculate by
 * brouCalculateMeanAndVariance3D
 *
 * the alpha and beta is knowed
 *
 * output = alpha * (input - mean) / (sqrt(variance + epsilon)) + beta
 */
/**
 * calculate the input mean and "varicance"
 */
kernel void BROU(CalculateMeanAndVariance3D)(device real *input             [[buffer(0)]],
                                             device real *mean              [[buffer(1)]],
                                             device real *variance          [[buffer(2)]],
                                             device float *e                [[buffer(3)]],
                                             constant TensorShape &shape    [[buffer(4)]],
                                             ushort grid [[thread_position_in_grid]]) {
    int height  = shape.dim0;
    int width   = shape.dim1;
    int channel = shape.dim2;
    
    int z = grid << 2;
    
    if (z >= channel) {
        return;
    }
    
    float epsilon = e[0];
    
    /**use float to store sum of input*/
    float4 sum = 0;
    
    /**calcualte mean*/
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            real4 inputV = ((device real4*)(input + (y * width + x) * channel + z))[0];
            
            sum += (static_cast<float4>(inputV));
        }
    }
    
    real4 meanV = static_cast<real4>(sum / (1.0 * height * width));
    
    /**calculate variance*/
    sum = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            real4 inputV = ((device real4*)(input + (y * width + x) * channel + z))[0];
            real4 differ = inputV - meanV;
            
            sum += static_cast<float4>(differ * differ);
        }
    }
    
    real4 varianceV = static_cast<real4>(1.0 / sqrt(sum / (1.0 * height * width) + epsilon));
    
    device real4 *mean4     = (device real4*)(mean     + z);
    device real4 *variance4 = (device real4*)(variance + z);
    
    mean4[0]     = meanV;
    variance4[0] = varianceV;
}

/**
 * every thread handle 4X4X4 block
 */
kernel void BROU(BatchNormalization3D)(device real *input           [[buffer(0)]],
                                       device real *output          [[buffer(1)]],
                                       device real *mean            [[buffer(2)]],
                                       device real *variance        [[buffer(3)]],
                                       device real *alpha           [[buffer(4)]],
                                       device real *beta            [[buffer(5)]],
                                       constant TensorShape &shape  [[buffer(6)]],
                                       ushort3 grid [[thread_position_in_grid]]) {
    int height  = shape.dim0;
    int width   = shape.dim1;
    int channel = shape.dim2;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (x >= width || y >= height || z >= channel) {
        return;
    }
    
    real4 meanV     = ((device real4*)(mean     + z))[0];
    real4 varianceV = ((device real4*)(variance + z))[0];
    real4 alphaV    = ((device real4*)(alpha    + z))[0];
    real4 betaV     = ((device real4*)(beta     + z))[0];
    
    int maxY = min(y + 4, height);
    int maxX = min(x + 4, width);
    
    for (int j = y; j < maxY; ++j) {
        for (int i = x; i < maxX; ++i) {
            device real4 *inputV  = (device real4*)(input  + (j * width + i) * channel + z);
            device real4 *outputV = (device real4*)(output + (j * width + i) * channel + z);
            
            outputV[0] = alphaV * (inputV[0] - meanV) * varianceV + betaV;
        }
    }
}

#endif












