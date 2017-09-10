/**
 * BrouMacro.metal
 * BrouhahaMetal
 *
 * Created by yanyuanchi on 2017/8/14.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * define some macro for the BrouhahaMetal
 * ref:Torch
 */

#include <metal_stdlib>
using namespace metal;

#define BROU_CONCAT_2_EXPAND(a, b) a ## b
#define BROU_CONCAT_2(a, b) BROU_CONCAT_2_EXPAND(a, b)

#define BROU_CONCAT_3_EXPAND(a, b, c) a ## b ## c
#define BROU_CONCAT_3(a, b, c) BROU_CONCAT_3_EXPAND(a, b, c)

#define BROU_CONCAT_4_EXPAND(a, b, c, d) a ## b ## c ## d
#define BROU_CONCAT_4(a, b, c, d) BROU_CONCAT_4_EXPAND(a, b, c, d)

#define BROU_CONCAT_5_EXPAND(a, b, c, d, e) a ## b ## c ## d ## e
#define BROU_CONCAT_5(a, b, c, d, e) BROU_CONCAT_5_EXPAND(a, b, c, d, e)

#define BROU(name) BROU_CONCAT_4(brou, name, _, real)

#define BROU_CONVERT(from, to, dim) BROU_CONCAT_5(brouConvertFrom, from, 2, to, dim)

/**use macro to do vector dot*/
#define VECTOR_DOT(a, offsetY, offsetX, b, y, x) dot(a[offsetY+y][offsetX+x], b[y][x])

#define ROW_VECTOR_DOT_COL0(a, offsetY, offsetX, b, row) VECTOR_DOT(a, offsetY, offsetX, b, row, 0)
#define ROW_VECTOR_DOT_COL1(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL0(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 1)
#define ROW_VECTOR_DOT_COL2(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL1(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 2)
#define ROW_VECTOR_DOT_COL3(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL2(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 3)
#define ROW_VECTOR_DOT_COL4(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL3(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 4)
#define ROW_VECTOR_DOT_COL5(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL4(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 5)
#define ROW_VECTOR_DOT_COL6(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL5(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 6)
#define ROW_VECTOR_DOT_COL7(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL6(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 7)
#define ROW_VECTOR_DOT_COL8(a, offsetY, offsetX, b, row) ROW_VECTOR_DOT_COL7(a, offsetY, offsetX, b, row)+VECTOR_DOT(a, offsetY, offsetX, b, row, 8)

#define COL_VECTOR_DOT_ROW0(a, offsetY, offsetX, b, col) VECTOR_DOT(a, offsetY, offsetX, b, 0, col)
#define COL_VECTOR_DOT_ROW1(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW0(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 1, col)
#define COL_VECTOR_DOT_ROW2(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW1(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 2, col)
#define COL_VECTOR_DOT_ROW3(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW2(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 3, col)
#define COL_VECTOR_DOT_ROW4(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW3(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 4, col)
#define COL_VECTOR_DOT_ROW5(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW4(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 5, col)
#define COL_VECTOR_DOT_ROW6(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW5(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 6, col)
#define COL_VECTOR_DOT_ROW7(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW6(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 7, col)
#define COL_VECTOR_DOT_ROW8(a, offsetY, offsetX, b, col) COL_VECTOR_DOT_ROW7(a, offsetY, offsetX, b, col)+VECTOR_DOT(a, offsetY, offsetX, b, 8, col)

#define MATRIX_VECTOR_DOT_0X0(a, offsetY, offsetX, b) VECTOR_DOT(a, offsetY, offsetX, b, 0, 0)
#define MATRIX_VECTOR_DOT_1X1(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_0X0(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL0(a, offsetY, offsetX, b, 1) \
                                                    + COL_VECTOR_DOT_ROW0(a, offsetY, offsetX, b, 1) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 1, 1)

#define MATRIX_VECTOR_DOT_2X2(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_1X1(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL1(a, offsetY, offsetX, b, 2) \
                                                    + COL_VECTOR_DOT_ROW1(a, offsetY, offsetX, b, 2) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 2, 2)

#define MATRIX_VECTOR_DOT_3X3(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_2X2(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL2(a, offsetY, offsetX, b, 3) \
                                                    + COL_VECTOR_DOT_ROW2(a, offsetY, offsetX, b, 3) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 3, 3)

#define MATRIX_VECTOR_DOT_4X4(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_3X3(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL3(a, offsetY, offsetX, b, 4) \
                                                    + COL_VECTOR_DOT_ROW3(a, offsetY, offsetX, b, 4) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 4, 4)

#define MATRIX_VECTOR_DOT_5X5(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_4X4(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL4(a, offsetY, offsetX, b, 5) \
                                                    + COL_VECTOR_DOT_ROW4(a, offsetY, offsetX, b, 5) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 5, 5)

#define MATRIX_VECTOR_DOT_6X6(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_5X5(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL5(a, offsetY, offsetX, b, 6) \
                                                    + COL_VECTOR_DOT_ROW5(a, offsetY, offsetX, b, 6) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 6, 6)

#define MATRIX_VECTOR_DOT_7X7(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_6X6(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL6(a, offsetY, offsetX, b, 7) \
                                                    + COL_VECTOR_DOT_ROW6(a, offsetY, offsetX, b, 7) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 7, 7)

#define MATRIX_VECTOR_DOT_8X8(a, offsetY, offsetX, b) MATRIX_VECTOR_DOT_7X7(a, offsetY, offsetX, b) \
                                                    + ROW_VECTOR_DOT_COL7(a, offsetY, offsetX, b, 8) \
                                                    + COL_VECTOR_DOT_ROW7(a, offsetY, offsetX, b, 8) \
                                                    + VECTOR_DOT(a, offsetY, offsetX, b, 8, 8)
