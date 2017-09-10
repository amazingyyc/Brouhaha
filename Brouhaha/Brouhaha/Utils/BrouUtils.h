#ifndef BrouUitls_h
#define BrouUitls_h

#include <stdio.h>
#include "BrouMacro.h"

#define type uint16_t
#define real half
#define real_is_half
#include "Generate/BrouGenerate.h"
#undef real
#undef real_is_half
#undef type

#define type float
#define real float
#define real_is_float
#include "Generate/BrouGenerate.h"
#undef real
#undef real_is_float
#undef type

#endif
