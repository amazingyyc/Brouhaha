/**
 * BrouPoolingLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the pooling layer
 */

#import <Foundation/Foundation.h>

#import "BrouLayer.h"

@interface BrouPoolingLayer : BrouLayer {
    /**
     * the input'd dimension is (inputHeight, inputWidth, channel)
     */
    int _inputHeight;
    int _inputWidth;

    /**
     * the input'd dimension is (outputHeight, outputWidth, channel)
     */
    int _outputWidth;
    int _outputHeight;

    int _channel;

    /**
     * the kernel window's size
     */
    int _kernelHeight;
    int _kernelWidth;

    /**the pad of the input*/
    int _padLeft;
    int _padTop;
    /**
     * the step
     */
    int _strideX;
    int _strideY;

    int _channelX4;
}

- (instancetype)initWithInputHeight:(int)inputHeight
                         inputWidth:(int)inputWidth
                       outputHeight:(int)outputHeight
                        outputWidth:(int)outputWidth
                            channel:(int)channel
                       kernelHeight:(int)kernelHeight
                        kernelWidth:(int)kerneWidth
                            padLeft:(int)padLeft
                             padTop:(int)padTop
                            strideX:(int)strideX
                            strideY:(int)strideY;

@end
