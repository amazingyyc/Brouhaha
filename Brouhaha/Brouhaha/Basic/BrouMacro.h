/**
 * ref: Torch
 */

#ifndef BrouMacro_h
#define BrouMacro_h

#define BROU_CONCAT_2_EXPAND(a, b) a ## b
#define BROU_CONCAT_2(a, b) BROU_CONCAT_2_EXPAND(a, b)

#define BROU_CONCAT_3_EXPAND(a, b, c) a ## b ## c
#define BROU_CONCAT_3(a, b, c) BROU_CONCAT_3_EXPAND(a, b, c)

#define BROU_CONCAT_4_EXPAND(a, b, c, d) a ## b ## c ## d
#define BROU_CONCAT_4(a, b, c, d) BROU_CONCAT_4_EXPAND(a, b, c, d)

#define BROU_CONCAT_5_EXPAND(a, b, c, d, e) a ## b ## c ## d ## e
#define BROU_CONCAT_5(a, b, c, d, e) BROU_CONCAT_5_EXPAND(a, b, c, d, e)

#define BROU_STR_EXPAND(name) #name
#define BROU_STR(name) BROU_STR_EXPAND(name)

#define BROU(name) BROU_CONCAT_4(brou, name, _, real)

#define BROU_OBJECT(name) BROU_CONCAT_4(Brou, name, _, real)
#define BROU_OBJECT_NAME(name) BROU_STR(BROU_CONCAT_4(Brou, name, _, real))

#define BROU_METAL(name) BROU_STR(BROU(name))

#define BROU_CONVERT_OBJECT(from, to) BROU_CONCAT_5(BrouConvertFrom, from, 2, to, Layer)
#define BROU_CONVERT_OBJECT_NAME(from, to) BROU_STR(BROU_CONCAT_5(BrouConvertFrom, from, 2, to, Layer))
#define BROU_CONVERT_METAL(from, to, dim) BROU_STR(BROU_CONCAT_5(brouConvertFrom, from, 2, to, dim))

#define BROU_MAX(a, b) (((a)>(b)) ? (a):(b))
#define BROU_MIN(a, b) (((a)>(b)) ? (b):(a))

#endif
