#import "Brouhaha.h"

#define type uint16_t
#define real half
#define real_is_half
#include "Generate/BrouConvolutionLayer.m"
#include "Generate/BrouConvolutionMMLayer.m"
#include "Generate/BrouTransposedConvolutionLayer.m"
#include "Generate/BrouTransposedConvolutionMMLayer.m"
#include "Generate/BrouDilatedConvolutionLayer.m"
#include "Generate/BrouDilatedConvolutionMMLayer.m"
#include "Generate/BrouPoolingLayer.m"
#include "Generate/BrouMaxPoolingLayer.m"
#include "Generate/BrouAveragePoolingLayer.m"
#include "Generate/BrouOperateLayer.m"
#include "Generate/BrouTanHLayer.m"
#include "Generate/BrouReLuLayer.m"
#include "Generate/BrouPReLuLayer.m"
#include "Generate/BrouLinearLayer.m"
#include "Generate/BrouFullConnectLayer.m"
#include "Generate/BrouAddLayer.m"
#include "Generate/BrouAddBiasLayer.m"
#include "Generate/BrouBatchNormalizationLayer.m"
#undef real
#undef real_is_half
#undef type

#define type float
#define real float
#define real_is_float
#include "Generate/BrouConvolutionLayer.m"
#include "Generate/BrouConvolutionMMLayer.m"
#include "Generate/BrouTransposedConvolutionLayer.m"
#include "Generate/BrouTransposedConvolutionMMLayer.m"
#include "Generate/BrouDilatedConvolutionLayer.m"
#include "Generate/BrouDilatedConvolutionMMLayer.m"
#include "Generate/BrouPoolingLayer.m"
#include "Generate/BrouMaxPoolingLayer.m"
#include "Generate/BrouAveragePoolingLayer.m"
#include "Generate/BrouOperateLayer.m"
#include "Generate/BrouTanHLayer.m"
#include "Generate/BrouReLuLayer.m"
#include "Generate/BrouPReLuLayer.m"
#include "Generate/BrouLinearLayer.m"
#include "Generate/BrouFullConnectLayer.m"
#include "Generate/BrouAddLayer.m"
#include "Generate/BrouAddBiasLayer.m"
#include "Generate/BrouBatchNormalizationLayer.m"
#undef real
#undef real_is_float
#undef type

/**
 * a convert layer used to convert number type
 */
#define from uchar
#define to half
#include "Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from half
#define to uchar
#include "Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from half
#define to float
#include "Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from float
#define to half
#include "Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from uchar
#define to float
#include "Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from float
#define to uchar
#include "Generate/BrouConvertLayer.m"
#undef to
#undef from


