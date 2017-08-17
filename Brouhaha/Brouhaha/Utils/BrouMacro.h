/**
 * BrouMacro.h
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/8/9.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#ifndef BrouMacro_h
#define BrouMacro_h

#define BROU_CONCAT_2_EXPAND(a, b) a##_##b
#define BROU_CONCAT_2(a, b) BROU_CONCAT_2_EXPAND(a, b)

#define BROU_NAME(name) BROU_CONCAT_2(name, real)

#define half uint16_t

#define BROU_MAX(a,b) (((a)>(b)) ? (a):(b))
#define BROU_MIN(a,b) (((a)>(b)) ? (b):(a))

#endif
