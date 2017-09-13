#ifndef BrouStruct_h
#define BrouStruct_h

/**
 * the data's dimension type
 */
typedef NS_ENUM(NSInteger, DimensionType) {
    Dimension1D = 1,
    Dimension2D = 2,
    Dimension3D = 3,
    Dimension4D = 4
};

/**
 * store the shpe of data
 * (dim0,dim1, dim2) represent a 3d dimension
 */
typedef struct _TensorShape {
    int32_t dim0;
    int32_t dim1;
    int32_t dim2;
    int32_t dim3;
} TensorShape;

/**
 * uae a struct to store the params of a convolution
 */
typedef struct _ConvolutionShape {
    /**the kernel size*/
    int32_t kernelHeight;
    int32_t kernelWidth;
    
    /**the pad of input*/
    int32_t padTop;
    int32_t padLeft;
    
    /**the stride of kernel, for transposed convolution always be 1*/
    int32_t strideY;
    int32_t strideX;
    
    /**the 0 units inserted to input of transposed convolution*/
    int32_t insertY;
    int32_t insertX;
    
    /**for dilated convolution*/
    int32_t dilatedY;
    int32_t dilatedX;
    
    /**if the convoluton has bias, 0 false, !0 true*/
    bool haveBias;
} ConvolutionShape;

/**
 * a struct to store the BatchNormalization params
 */
typedef struct _BatchNormalizationShape {
    /**the epsilon of BN*/
    float epsilon;
    
    /**every thread deal with (perThreadWidth, perThreadHeight) input*/
    int32_t perThreadWidth;
    int32_t perThreadHeight;
} BatchNormalizationShape;

#endif
