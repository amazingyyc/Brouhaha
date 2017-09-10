/**
 * BrouhahaMetal
 *
 * Created by yanyuanchi on 2017/8/14.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * ref: Torch
 */

#include <metal_stdlib>

#include "BrouMacro.metal"
#include "BrouStruct.metal"

using namespace metal;

#define real half
#define real4 half4
#define real_is_half

#include "BrouTanH.metal"
#include "BrouAdd.metal"
#include "BrouAddBias.metal"
#include "BrouMaxPooling.metal"
#include "BrouAveragePooling.metal"
#include "BrouMatrixMultiply.metal"
#include "BrouConvolution.metal"
#include "BrouConvolutionMM.metal"
#include "BrouTransposedConvolution.metal"
#include "BrouTransposedConvolutionMM.metal"
#include "BrouDilatedConvolution.metal"
#include "BrouDilatedConvolutionMM.metal"
#include "BrouFullconnect.metal"
#include "BrouLinear.metal"
#include "BrouReLu.metal"
#include "BrouPReLu.metal"
#include "BrouBatchNormalization.metal"

#undef real_is_half
#undef real4
#undef real

#define real float
#define real4 float4
#define real_is_float

#include "BrouTanH.metal"
#include "BrouAdd.metal"
#include "BrouAddBias.metal"
#include "BrouMaxPooling.metal"
#include "BrouAveragePooling.metal"
#include "BrouMatrixMultiply.metal"
#include "BrouConvolution.metal"
#include "BrouConvolutionMM.metal"
#include "BrouTransposedConvolution.metal"
#include "BrouTransposedConvolutionMM.metal"
#include "BrouDilatedConvolution.metal"
#include "BrouDilatedConvolutionMM.metal"
#include "BrouFullconnect.metal"
#include "BrouLinear.metal"
#include "BrouReLu.metal"
#include "BrouPReLu.metal"
#include "BrouBatchNormalization.metal"

#undef real_is_float
#undef real4
#undef real

#define from uchar
#define from4 uchar4
#define to half
#define to4 half4
#include "BrouConvert.metal"
#undef to4
#undef to
#undef from4
#undef from

#define from half
#define from4 half4
#define to uchar
#define to4 uchar4
#include "BrouConvert.metal"
#undef to4
#undef to
#undef from4
#undef from

#define from uchar
#define from4 uchar4
#define to float
#define to4 float4
#include "BrouConvert.metal"
#undef to4
#undef to
#undef from4
#undef from

#define from float
#define from4 float4
#define to uchar
#define to4 uchar4
#include "BrouConvert.metal"
#undef to4
#undef to
#undef from4
#undef from

#define from half
#define from4 half4
#define to float
#define to4 float4
#include "BrouConvert.metal"
#undef to4
#undef to
#undef from4
#undef from

#define from float
#define from4 float4
#define to half
#define to4 half4
#include "BrouConvert.metal"
#undef to4
#undef to
#undef from4
#undef from










