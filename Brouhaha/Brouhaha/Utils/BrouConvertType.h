#ifndef BrouConvertType_h
#define BrouConvertType_h

#include <stdio.h>

/**
 * convert float and half
 */
void convertFloat16ToFloat32(uint16_t *half, uint32_t *single, int length);
void convertFloat32ToFloat16(uint32_t *single, uint16_t *half, int length);
void convertFloat32ToFloat16Two(uint32_t *s1, uint16_t *h1, int l1, uint32_t *s2, uint16_t *h2, int l2);

uint16_t convertFloat32ToFloat16OneNumber(uint32_t *single);

#endif
