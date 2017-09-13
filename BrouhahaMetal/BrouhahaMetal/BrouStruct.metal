/**
 * define the structs
 *
 * Created by yanyuanchi on 2017/7/23.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#include <metal_stdlib>
using namespace metal;

/**
 * a tensor dimension struct
 * (dim0 dim1 dim2) represent a 3d tensor
 */
typedef struct {
    int dim0;
    int dim1;
    int dim2;
    int dim3;
} TensorShape;

/**
 * uae a struct to store the params of a convolution
 */
typedef struct {
    /**the kernel size*/
    int kernelHeight;
    int kernelWidth;
    
    /**the pad of input*/
    int padTop;
    int padLeft;
    
    /**the stride of kernel, for transposed convolution always be 1*/
    int strideY;
    int strideX;
    
    /**the 0 units inserted to input of transposed convolution*/
    int insertY;
    int insertX;
    
    /**for dilated convolution*/
    int dilatedY;
    int dilatedX;
    
    /**if the convoluton has bias*/
    bool haveBias;
} ConvolutionShape;

/**
 * a struct to store the BatchNormalization params
 */
typedef struct {
    /**the epsilon of BN*/
    float epsilon;
    
    /**every thread deal with (perThreadWidth, perThreadHeight) input*/
    int perThreadWidth;
    int perThreadHeight;
} BatchNormalizationShape;









