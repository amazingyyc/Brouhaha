/**
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 * transposedconvolutionlayer need 2 temp matrix to store the ephemeral data,
 * so it may be out of the limit memory of MTLBuffer(256MB)
 * like a input (512, 512, 32) output(1024, 1024, 32) kernel(32, 9, 9, 32)
 * for half it needs 32 * 9 * 9 * 512 * 512 * 2 byte = 1296MB, bigger than 256MB
 * so if the image is big, the BrouTransposedConvolutionLayer is better choice
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#if defined(real) && defined(real4) && defined(BROU)

/**
 * convert the input to a matrix
 * the input dimension is (inputHeight, inputWidth, inputChannelX4)
 * looks as a matrix (inputHeight*inputWidth, inputChannelX4)
 * then the output matrix is the transposed of input and the dimension is (inputChannelX4, [inputHeight*inputWidth]X4)
 *
 * the  shape.dim0 = inputHeight*inputWidth shape.dim1 = inputChannelX4 shape.dim2 = inputHeight*inputWidthX4
 * every thread deal with 4X4 block
 */
kernel void BROU(TransposedConvolutionInput2Matrix)(device real *input          [[buffer(0)]],
                                                    device real *matrix         [[buffer(1)]],
                                                    constant TensorShape& shape [[buffer(2)]],
                                                    ushort2 grid [[thread_position_in_grid]]) {
    int row = grid.y << 2;
    int col = grid.x << 2;
    
    int nx4 = shape.dim1;
    int mx4 = shape.dim2;
    
    if (row >= nx4 || col >= mx4) {
        return;
    }
    
    int m = shape.dim0;
    
    real4 in0 = (col >= m)       ? 0 : ((device real4*)(input + col       * nx4 + row))[0];
    real4 in1 = ((col + 1) >= m) ? 0 : ((device real4*)(input + (col + 1) * nx4 + row))[0];
    real4 in2 = ((col + 2) >= m) ? 0 : ((device real4*)(input + (col + 2) * nx4 + row))[0];
    real4 in3 = ((col + 3) >= m) ? 0 : ((device real4*)(input + (col + 3) * nx4 + row))[0];
    
    device real4 *matrix0 = (device real4*)(matrix +  row      * mx4 + col);
    device real4 *matrix1 = (device real4*)(matrix + (row + 1) * mx4 + col);
    device real4 *matrix2 = (device real4*)(matrix + (row + 2) * mx4 + col);
    device real4 *matrix3 = (device real4*)(matrix + (row + 3) * mx4 + col);
    
    matrix0[0] = {in0.x, in1.x, in2.x, in3.x};
    matrix1[0] = {in0.y, in1.y, in2.y, in3.y};
    matrix2[0] = {in0.z, in1.z, in2.z, in3.z};
    matrix3[0] = {in0.w, in1.w, in2.w, in3.w};
}

/**
 * convert the matrxi to output
 * the matrix dimension is (outputChannelX4*kernelHeight*kernelWidth, [inputHeight*inputWidth]X4)
 * every thread output 4X4 block
 */
kernel void BROU(TransposedConvolutionMatrix2Output)(device real *matrix               [[buffer(0)]],
                                                     device real *output               [[buffer(1)]],
                                                     device real *bia                  [[buffer(2)]],
                                                     constant TensorShape& matrixShape [[buffer(3)]],
                                                     constant TensorShape& inputShape  [[buffer(4)]],
                                                     constant TensorShape& outputShape [[buffer(5)]],
                                                     constant ConvolutionShape& convolutionShape [[buffer(6)]],
                                                     ushort3 grid [[thread_position_in_grid]]) {
    int y = grid.y << 2;
    int x = grid.x << 2;
    int z = grid.z;
    
    int outputHeight  = outputShape.dim0;
    int outputWidth   = outputShape.dim1;
    int outputChannel = outputShape.dim2;
    
    if (y >= outputHeight || x >= outputWidth || z >= outputChannel) {
        return;
    }
    
    // int m = matrixShape.dim0;
    int n = matrixShape.dim1;
    
    int inputHeight  = inputShape.dim0;
    int inputWidth   = inputShape.dim1;
    
    int kernelHeight = convolutionShape.kernelHeight;
    int kernelWidth  = convolutionShape.kernelWidth;
    
    int padTop  = convolutionShape.padTop;
    int padLeft = convolutionShape.padLeft;
    
    int insertY = convolutionShape.insertY;
    int insertX = convolutionShape.insertX;
    
    int insertYAdd1 = insertY + 1;
    int insertXAdd1 = insertX + 1;
    
    int fakeInputHeight = inputHeight + (inputHeight - 1) * insertY;
    int fakeInputWidth  = inputWidth  + (inputWidth  - 1) * insertX;
    
    real biasV = convolutionShape.haveBias ? (bia + z)[0] : 0;
    
    int maxOutY = min(y + 4, outputHeight);
    int maxOutX = min(x + 4, outputWidth);
    
    int rowOffset = z * kernelHeight * kernelWidth;
    
    for (int outY = y; outY < maxOutY; ++outY) {
        for (int outX = x; outX < maxOutX; ++outX) {
            /**store the out*/
            real out = biasV;
            
            int inputTop  = -padTop  + outY;
            int inputLeft = -padLeft + outX;
            
            int inputBottom = min(inputTop  + kernelHeight, fakeInputHeight);
            int inputRight  = min(inputLeft + kernelWidth ,  fakeInputWidth);
            
            int realInputTop  = (0 > inputTop)  ? 0 : ((inputTop  + insertY) / insertYAdd1 * insertYAdd1);
            int realInputLeft = (0 > inputLeft) ? 0 : ((inputLeft + insertX) / insertXAdd1 * insertXAdd1);
            
            int kernelTop  = realInputTop  - inputTop;
            int kernelLeft = realInputLeft - inputLeft;
            
            for (int inY = realInputTop, kernelY = kernelTop; inY < inputBottom; inY += insertYAdd1, kernelY += insertYAdd1) {
                for (int inX = realInputLeft, kernelX = kernelLeft; inX < inputRight; inX += insertXAdd1, kernelX += insertXAdd1) {
                    int realInY = inY / insertYAdd1;
                    int realInX = inX / insertXAdd1;
                    
                    out += matrix[(rowOffset + kernelY * kernelWidth + kernelX) * n + realInY * inputWidth + realInX];
                }
            }
            
            device real *outputV = output + (outY * outputWidth + outX) * outputChannel + z;
            outputV[0] = out;
        }
    }
}

#endif




















