/**
 * BrouFullConnectLayer.h
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the fullconnect layer
 */

#import "BrouLayer.h"

@interface BrouFullConnectLayer : BrouLayer

- (instancetype)initWithFloat32Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                              weights:(void*)weightsData
                                 bias:(void*)biasData
                        intputChannel:(int)inputChannel
                        outputChannel:(int)outputChannel;

- (instancetype)initWithFloat16Device:(id<MTLDevice>)device
                              library:(id<MTLLibrary>)library
                              weights:(void*)weightsData
                                 bias:(void*)biasData
                        intputChannel:(int)inputChannel
                        outputChannel:(int)outputChannel;

@end
