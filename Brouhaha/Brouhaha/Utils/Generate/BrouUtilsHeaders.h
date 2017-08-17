/**
 * Brouhaha
 *
 * Created by yanyuanchi on 2017/8/12.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 */

#if defined(real) && defined(BROU_NAME)

/**
 * transpose the in matrix to out
 * requirt:outRow >= inCol outCol >= inRow
 */
void BROU_NAME(brouTransposeMatrix)(real *in, size_t inRow, size_t inCol, real *out, size_t outRow, size_t outCol);

#endif
