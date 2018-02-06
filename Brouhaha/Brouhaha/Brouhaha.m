#import "Brouhaha.h"

#define type uint16_t
#define real half
#define real_is_half

#include "Basic/BrouUniqueTensor.m"
#include "Basic/BrouTemporaryTensor.m"

#include "Layer/Generate/BrouConvolutionLayer.m"
#include "Layer/Generate/BrouConvolutionMMLayer.m"
#include "Layer/Generate/BrouTransposedConvolutionLayer.m"
#include "Layer/Generate/BrouTransposedConvolutionMMLayer.m"
#include "Layer/Generate/BrouDilatedConvolutionLayer.m"
#include "Layer/Generate/BrouDilatedConvolutionMMLayer.m"
#include "Layer/Generate/BrouMaxPoolingLayer.m"
#include "Layer/Generate/BrouAveragePoolingLayer.m"
#include "Layer/Generate/BrouTanHLayer.m"
#include "Layer/Generate/BrouReLuLayer.m"
#include "Layer/Generate/BrouPReLuLayer.m"
#include "Layer/Generate/BrouLinearLayer.m"
#include "Layer/Generate/BrouFullConnectLayer.m"
#include "Layer/Generate/BrouAddLayer.m"
#include "Layer/Generate/BrouAddBiasLayer.m"
#include "Layer/Generate/BrouBatchNormalizationLayer.m"
#undef real
#undef real_is_half
#undef type

#define type float
#define real float
#define real_is_float

#include "Basic/BrouUniqueTensor.m"
#include "Basic/BrouTemporaryTensor.m"

#include "Layer/Generate/BrouConvolutionLayer.m"
#include "Layer/Generate/BrouConvolutionMMLayer.m"
#include "Layer/Generate/BrouTransposedConvolutionLayer.m"
#include "Layer/Generate/BrouTransposedConvolutionMMLayer.m"
#include "Layer/Generate/BrouDilatedConvolutionLayer.m"
#include "Layer/Generate/BrouDilatedConvolutionMMLayer.m"
#include "Layer/Generate/BrouMaxPoolingLayer.m"
#include "Layer/Generate/BrouAveragePoolingLayer.m"
#include "Layer/Generate/BrouTanHLayer.m"
#include "Layer/Generate/BrouReLuLayer.m"
#include "Layer/Generate/BrouPReLuLayer.m"
#include "Layer/Generate/BrouLinearLayer.m"
#include "Layer/Generate/BrouFullConnectLayer.m"
#include "Layer/Generate/BrouAddLayer.m"
#include "Layer/Generate/BrouAddBiasLayer.m"
#include "Layer/Generate/BrouBatchNormalizationLayer.m"
#undef real
#undef real_is_float
#undef type

/**
 * a convert layer used to convert number type
 */
#define from uchar
#define to half
#include "Layer/Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from half
#define to uchar
#include "Layer/Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from half
#define to float
#include "Layer/Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from float
#define to half
#include "Layer/Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from uchar
#define to float
#include "Layer/Generate/BrouConvertLayer.m"
#undef to
#undef from

#define from float
#define to uchar
#include "Layer/Generate/BrouConvertLayer.m"
#undef to
#undef from


