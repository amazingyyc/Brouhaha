#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "BrouMacro.h"
#import "BrouStruct.h"
#import "BrouUtils.h"
#import "BrouConvertType.h"
#import "BrouLayer.h"

#define type uint16_t
#define real half
#define real_is_half
#include "Generate/BrouConvolutionLayer.h"
#include "Generate/BrouConvolutionMMLayer.h"
#include "Generate/BrouTransposedConvolutionLayer.h"
#include "Generate/BrouTransposedConvolutionMMLayer.h"
#include "Generate/BrouDilatedConvolutionLayer.h"
#include "Generate/BrouDilatedConvolutionMMLayer.h"
#include "Generate/BrouPoolingLayer.h"
#include "Generate/BrouMaxPoolingLayer.h"
#include "Generate/BrouAveragePoolingLayer.h"
#include "Generate/BrouOperateLayer.h"
#include "Generate/BrouTanHLayer.h"
#include "Generate/BrouReLuLayer.h"
#include "Generate/BrouPReLuLayer.h"
#include "Generate/BrouLinearLayer.h"
#include "Generate/BrouFullConnectLayer.h"
#include "Generate/BrouAddLayer.h"
#include "Generate/BrouAddBiasLayer.h"
#include "Generate/BrouBatchNormalizationLayer.h"
#undef real
#undef real_is_half
#undef type

#define type float
#define real float
#define real_is_float
#include "Generate/BrouConvolutionLayer.h"
#include "Generate/BrouConvolutionMMLayer.h"
#include "Generate/BrouTransposedConvolutionLayer.h"
#include "Generate/BrouTransposedConvolutionMMLayer.h"
#include "Generate/BrouDilatedConvolutionLayer.h"
#include "Generate/BrouDilatedConvolutionMMLayer.h"
#include "Generate/BrouPoolingLayer.h"
#include "Generate/BrouMaxPoolingLayer.h"
#include "Generate/BrouAveragePoolingLayer.h"
#include "Generate/BrouOperateLayer.h"
#include "Generate/BrouTanHLayer.h"
#include "Generate/BrouReLuLayer.h"
#include "Generate/BrouPReLuLayer.h"
#include "Generate/BrouLinearLayer.h"
#include "Generate/BrouFullConnectLayer.h"
#include "Generate/BrouAddLayer.h"
#include "Generate/BrouAddBiasLayer.h"
#include "Generate/BrouBatchNormalizationLayer.h"
#undef real
#undef real_is_float
#undef type

/**
 * a convert layer used to convert number type
 */
#define from uchar
#define to half
#include "Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from half
#define to uchar
#include "Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from half
#define to float
#include "Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from float
#define to half
#include "Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from uchar
#define to float
#include "Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from float
#define to uchar
#include "Generate/BrouConvertLayer.h"
#undef to
#undef from



