#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "BrouMacro.h"
#import "BrouStruct.h"
#import "BrouUtils.h"
#import "BrouConvertType.h"
#import "BrouShareBuffer.h"
#import "BrouTensor.h"
#import "BrouLayer.h"

#define type uint16_t
#define real half
#define real_is_half

#include "Basic/BrouUniqueTensor.h"
#include "Basic/BrouTemporaryTensor.h"

#include "Layer/Generate/BrouConvolutionLayer.h"
#include "Layer/Generate/BrouConvolutionMMLayer.h"
#include "Layer/Generate/BrouTransposedConvolutionLayer.h"
#include "Layer/Generate/BrouTransposedConvolutionMMLayer.h"
#include "Layer/Generate/BrouDilatedConvolutionLayer.h"
#include "Layer/Generate/BrouDilatedConvolutionMMLayer.h"
#include "Layer/Generate/BrouMaxPoolingLayer.h"
#include "Layer/Generate/BrouAveragePoolingLayer.h"
#include "Layer/Generate/BrouTanHLayer.h"
#include "Layer/Generate/BrouReLuLayer.h"
#include "Layer/Generate/BrouPReLuLayer.h"
#include "Layer/Generate/BrouLinearLayer.h"
#include "Layer/Generate/BrouFullConnectLayer.h"
#include "Layer/Generate/BrouAddLayer.h"
#include "Layer/Generate/BrouAddBiasLayer.h"
#include "Layer/Generate/BrouBatchNormalizationLayer.h"
#undef real
#undef real_is_half
#undef type

#define type float
#define real float
#define real_is_float

#include "Basic/BrouUniqueTensor.h"
#include "Basic/BrouTemporaryTensor.h"

#include "Layer/Generate/BrouConvolutionLayer.h"
#include "Layer/Generate/BrouConvolutionMMLayer.h"
#include "Layer/Generate/BrouTransposedConvolutionLayer.h"
#include "Layer/Generate/BrouTransposedConvolutionMMLayer.h"
#include "Layer/Generate/BrouDilatedConvolutionLayer.h"
#include "Layer/Generate/BrouDilatedConvolutionMMLayer.h"
#include "Layer/Generate/BrouMaxPoolingLayer.h"
#include "Layer/Generate/BrouAveragePoolingLayer.h"
#include "Layer/Generate/BrouTanHLayer.h"
#include "Layer/Generate/BrouReLuLayer.h"
#include "Layer/Generate/BrouPReLuLayer.h"
#include "Layer/Generate/BrouLinearLayer.h"
#include "Layer/Generate/BrouFullConnectLayer.h"
#include "Layer/Generate/BrouAddLayer.h"
#include "Layer/Generate/BrouAddBiasLayer.h"
#include "Layer/Generate/BrouBatchNormalizationLayer.h"
#undef real
#undef real_is_float
#undef type

/**
 * a convert layer used to convert number type
 */
#define from uchar
#define to half
#include "Layer/Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from half
#define to uchar
#include "Layer/Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from half
#define to float
#include "Layer/Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from float
#define to half
#include "Layer/Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from uchar
#define to float
#include "Layer/Generate/BrouConvertLayer.h"
#undef to
#undef from

#define from float
#define to uchar
#include "Layer/Generate/BrouConvertLayer.h"
#undef to
#undef from



