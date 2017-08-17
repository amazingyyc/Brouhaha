/**
 * BrouPoolingLayer
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the pooling layer
 */

#import "BrouPoolingLayer.h"

@interface BrouPoolingLayer() {
}

@end

@implementation BrouPoolingLayer

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
                            strideY:(int)strideY {
    self = [super initWithLabel:@"BrouPoolingLayer"];

    if (!self) {
        return self;
    }

    _inputHeight = inputHeight;
    _inputWidth  = inputWidth;

    _outputHeight = outputHeight;
    _outputWidth  = outputWidth;

    _channel = channel;

    _kernelHeight = kernelHeight;
    _kernelWidth  = kerneWidth;

    _padLeft = padLeft;
    _padTop  = padTop;

    _strideX = strideX;
    _strideY = strideY;

    _channelX4 = (_channel + 3) / 4 * 4;

    return self;
}

@end











