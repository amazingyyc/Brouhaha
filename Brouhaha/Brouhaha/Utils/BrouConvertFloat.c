/**
 * utils.c
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/5/17.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 * 
 * ref:Fast Half Float Conversions Jeroen van der Zijp November 2008 (Revised September 2010)
 */

#include <stdio.h>

#include "BrouConvertFloat.h"

uint32_t convertmantissa(uint32_t i) {
    uint32_t m = i << 13;
    uint32_t e = 0;

    while (!(m & 0x00800000)) {
        e -= 0x00800000;
        m <<= 1;
    }

    m &= ~(0x00800000);
    e += 0x38800000;

    return m | e;
}

/**
 * convert the float16 to float32
 * ref:Fast Half Float Conversions Jeroen van der Zijp November 2008 (Revised September 2010)
 */
void convertFloat16ToFloat32(uint16_t *half, uint32_t *single, int length) {
    uint32_t exponenttable[64];
    uint32_t offsettable[64];
    uint32_t mantissatable[2048];

    exponenttable[0]  = 0;
    exponenttable[32] = 0x80000000;
    exponenttable[31] = 0x47800000;
    exponenttable[63] = 0xC7800000;

    for (uint32_t i = 1; i <= 30; ++i) {
        exponenttable[i] = i << 23;
    }

    for (uint32_t i = 33; i <= 62; ++i) {
        exponenttable[i] = 0x80000000 + ((i - 32) << 23);
    }

    for (uint32_t i = 0; i < 64; ++i) {
        offsettable[i] = 1024;
    }

    offsettable[0]  = 0;
    offsettable[32] = 0;

    mantissatable[0] = 0;

    for (int i = 1; i < 1024; ++i) {
        mantissatable[i] = convertmantissa(i);
    }

    for (int i = 1024; i < 2048; ++i) {
        mantissatable[i] = 0x38000000 + ((i - 1024) << 13);
    }

    int limit = length - 3;
    int i = 0;
    for (; i < limit; i += 4) {
        single[i]   = mantissatable[offsettable[half[i] >> 10] + (half[i] & 0x3ff)] + exponenttable[half[i] >> 10];
        single[i+1] = mantissatable[offsettable[half[i+1] >> 10] + (half[i+1] & 0x3ff)] + exponenttable[half[i+1] >> 10];
        single[i+2] = mantissatable[offsettable[half[i+2] >> 10] + (half[i+2] & 0x3ff)] + exponenttable[half[i+2] >> 10];
        single[i+3] = mantissatable[offsettable[half[i+3] >> 10] + (half[i+3] & 0x3ff)] + exponenttable[half[i+3] >> 10];
    }

    for (; i < length; ++i) {
        single[i] = mantissatable[offsettable[half[i] >> 10] + (half[i] & 0x3ff)] + exponenttable[half[i] >> 10];
    }
}

void initFloat32ToFloat16Table(uint16_t *basetable, int16_t *shifttable) {
    int e;
    for (uint32_t i = 0; i < 256; ++i) {
        e = i - 127;
        if(e < -24) {
            basetable[i|0x000]=0x0000;
            basetable[i|0x100]=0x8000;
            shifttable[i|0x000]=24;
            shifttable[i|0x100]=24;
        } else if(e < -14) {
            basetable[i|0x000]=(0x0400>>(-e-14));
            basetable[i|0x100]=(0x0400>>(-e-14)) | 0x8000;
            shifttable[i|0x000]=-e-1;
            shifttable[i|0x100]=-e-1;
        } else if(e <= 15) {
            basetable[i|0x000]=((e+15)<<10);
            basetable[i|0x100]=((e+15)<<10) | 0x8000;
            shifttable[i|0x000]=13;
            shifttable[i|0x100]=13;
        } else if(e < 128) {
            basetable[i|0x000]=0x7C00;
            basetable[i|0x100]=0xFC00;
            shifttable[i|0x000]=24;
            shifttable[i|0x100]=24;
        } else {
            basetable[i|0x000]=0x7C00;
            basetable[i|0x100]=0xFC00;
            shifttable[i|0x000]=13;
            shifttable[i|0x100]=13;
        }
    }
}

void convertFloat32ToFloat16WithTable(uint32_t *single, uint16_t *half, int length,
                                      uint16_t *basetable, int16_t *shifttable) {
    int limit = length - 3;
    int i = 0;

    for (; i < limit; i += 4) {
        half[i]   = basetable[(single[i] >> 23)&0x1ff] + ((single[i]   & 0x007fffff) >> shifttable[(single[i]   >> 23) & 0x1ff]);
        half[i+1] = basetable[(single[i+1]>>23)&0x1ff] + ((single[i+1] & 0x007fffff) >> shifttable[(single[i+1] >> 23) & 0x1ff]);
        half[i+2] = basetable[(single[i+2]>>23)&0x1ff] + ((single[i+2] & 0x007fffff) >> shifttable[(single[i+2] >> 23) & 0x1ff]);
        half[i+3] = basetable[(single[i+3]>>23)&0x1ff] + ((single[i+3] & 0x007fffff) >> shifttable[(single[i+3] >> 23) & 0x1ff]);
    }

    for (; i < length; ++i) {
        half[i] = basetable[(single[i] >> 23) & 0x1ff] + ((single[i] & 0x007fffff) >> shifttable[(single[i] >> 23) & 0x1ff]);
    }
}

/**
 * convert float32 to float16
 */
void convertFloat32ToFloat16(uint32_t *single, uint16_t *half, int length) {
    uint16_t basetable[512];
    int16_t shifttable[512];

    initFloat32ToFloat16Table(basetable,  shifttable);
    convertFloat32ToFloat16WithTable(single, half, length, basetable, shifttable);
}

void convertFloat32ToFloat16Two(uint32_t *s1, uint16_t *h1, int l1, uint32_t *s2, uint16_t *h2, int l2) {
    uint16_t basetable[512];
    int16_t shifttable[512];

    initFloat32ToFloat16Table(basetable,  shifttable);
    convertFloat32ToFloat16WithTable(s1, h1, l1, basetable, shifttable);
    convertFloat32ToFloat16WithTable(s2, h2, l2, basetable, shifttable);
}

uint16_t convertFloat32ToFloat16OneNumber(uint32_t *single) {
    uint32_t f = single[0];

    int index = (f >> 23) & 0x1ff;
    int i = index & 0xff;
    int e = i -127;

    uint16_t base;
    uint16_t shift;
    if (e < -24) {
        base  = ((index >> 8) & 1) ? 0x8000 : 0x0000;
        shift = 24;
    } else if (e < -14) {
        base  = ((index >> 8) & 1) ? ((0x0400>>(-e-14)) | 0x8000) : (0x0400>>(-e-14));
        shift = -e - 1;
    } else if (e <= 15) {
        base  = ((index >> 8) & 1) ? (((e+15)<<10) | 0x8000) : ((e+15)<<10);
        shift = 13;
    } else if (e < 128) {
        base  = ((index >> 8) & 1) ? 0xFC00 : 0x7C00;
        shift = 24;
    } else {
        base  = ((index >> 8) & 1) ? 0xFC00 : 0x7C00;
        shift = 13;
    }

    return base + ((f & 0x007fffff) >> shift);
}
















