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
    int32_t dim0;
    int32_t dim1;
    int32_t dim2;
} TensorShape;

/**
 * uae a struct to store the params of a convolution
 */
typedef struct {
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
    
    /**if the convoluton has bias*/
    bool haveBias;
} ConvolutionShape;











