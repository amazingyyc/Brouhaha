/**
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/7/30.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#import "BrouLayer.h"
#import "BrouConvolutionLayer.h"

@interface BrouTransposedConvolutionLayer : BrouConvolutionLayer {
    /**
     * the transposed convolution has a coperated convolution
     * the prefix _origin means the coperated convolution (origin convolution)
     * the property that without _origin mean the transpoed convolution's property
     */
    
    /**
     * the origin convolution input dimension
     */
    int _originInputHeight;
    int _originInputWidth;
    int _originInputChannel;
    
    /**
     * the origin convolution output dimension
     */
    int _originOutputeHeight;
    int _originOutputWidth;
    int _originOutputChannnel;
    
    /**
     * the origin convoluton's kernel
     */
    int _originKernelHeight;
    int _originKernelWidth;
    
    /**
     * the origin pad
     */
    int _originPadLeft;
    int _originPadTop;
    
    /**
     * the origin stride
     */
    int _originStrideX;
    int _originStrideY;
    
    /**
     * insertX = strideX - 1
     * insertY = strideY - 1
     * insert 0-uints to the input on x/y axis
     */
    int _insertX;
    int _insertY;
    
    /**
     * ref:A guide to convolution arithmetic for deep learning
     *
     * add to the output, default is 0, can not greater than stride - 1
     * outputWidth  = (intputWidth -  1) * strideX + kernelWidth  - 2 * padLeft + outputAddRight
     * outputHeight = (intputHeight - 1) * strideY + kernelHeight - 2 * padTop  + outputAddBottom
     */
    int _outputAddRight;
    int _outputAddBottom;
    
    /**store the input ouput convolution params*/
    id<MTLBuffer> _inputShape;
    id<MTLBuffer> _outputShape;
    
    id<MTLBuffer> _convolutionShape;
}

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                               kernel:(void*)kernelData
                                 bias:(void*)biasData
                    originInputHeight:(int)originInputHeight
                     originInputWidth:(int)originInputWidth
                   originInputChannel:(int)originInputChannel
                   originOutputHeight:(int)originOutputHeight
                    originOutputWidth:(int)originOutputWidth
                  originOutputChannel:(int)originOutputChannel
                   originKernelHeight:(int)originKernelHeight
                    originKernelWidth:(int)originKernelWidth
                        originPadLeft:(int)originPadLeft
                         originPadTop:(int)originPadTop
                        originStrideX:(int)originStrideX
                        originStrideY:(int)originStrideY
                       outputAddRight:(int)outputAddRight
                      outputAddBottom:(int)outputAddBottom;

- (void)configParametersOriginInputHeight:(int)originInputHeight
                         originInputWidth:(int)originInputWidth
                       originInputChannel:(int)originInputChannel
                       originOutputHeight:(int)originOutputHeight
                        originOutputWidth:(int)originOutputWidth
                      originOutputChannel:(int)originOutputChannel
                       originKernelHeight:(int)originKernelHeight
                        originKernelWidth:(int)originKernelWidth
                            originPadLeft:(int)originPadLeft
                             originPadTop:(int)originPadTop
                            originStrideX:(int)originStrideX
                            originStrideY:(int)originStrideY
                           outputAddRight:(int)outputAddRight
                          outputAddBottom:(int)outputAddBottom;
@end
