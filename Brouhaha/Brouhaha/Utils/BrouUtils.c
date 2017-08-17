/**
 * BrouUtils.c
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/8/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#include <stdio.h>

#include "BrouMacro.h"
#include "BrouNeon.h"

#define real half
#define real_is_half

#include "Generate/BrouMatrix.c"

#undef real
#undef real_is_half
