#include <stdio.h>

#include "BrouMacro.h"
#include "BrouNeon.c"

#define type uint16_t
#define real half
#define real_is_half
#include "Generate/BrouMatrix.c"
#undef real
#undef real_is_half
#undef type

#define type float
#define real float
#define real_is_float
#include "Generate/BrouMatrix.c"
#undef real
#undef real_is_float
#undef type
