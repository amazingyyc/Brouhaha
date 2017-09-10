#if defined(type) && defined(real) && defined(BROU)

/**
 * transpose the in matrix to out
 * requirt:outRow >= inCol outCol >= inRow
 */
void BROU(TransposeMatrix)(type *in, size_t inRow, size_t inCol, type *out, size_t outRow, size_t outCol);

#endif

