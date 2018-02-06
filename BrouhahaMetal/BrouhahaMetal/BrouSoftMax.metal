#if defined(real) && defined(real4) && defined(BROU)

/**
 for 1D softmax, a thread group contains 32 threads
 the output channel will be diviede into 32 blocks
 one thread output a block
 the input/output's dimension is (channel)
 intput: the input data
 output: the output data
 shape:
 shape.dim0 is smallest number that not less than channel and must be divided by 4 without remainder
 shape dim1 is channel of input/output
 
 creat 1 thread group and 32 threads in a group
 */
kernel void BROU(SoftMax1D)(device real *input           [[buffer(0)]],
                            device real *output          [[buffer(1)]],
                            constant TensorShape& shape  [[buffer(2)]],
                            ushort index [[thread_index_in_threadgroup]]) {
    int channel   = shape.dim1;
    int blockSize = (channel + 31) / 32;
    
    int minI = index * blockSize;
    int maxI = min((index + 1) * blockSize, channel);
    
    threadgroup real sum = 0;
    threadgroup real sharedSum[32];
    
    real curSum = 0;
    for (int i = minI; i < maxI; ++i) {
        real e = exp(input[i]);
        output[i] = e;
        
        curSum += e;
    }
    
    sharedSum[index] = curSum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (0 == index) {
        sum = (sharedSum[0]  + sharedSum[1]  + sharedSum[2]  + sharedSum[3] +
               sharedSum[4]  + sharedSum[5]  + sharedSum[6]  + sharedSum[7] +
               sharedSum[8]  + sharedSum[9]  + sharedSum[10] + sharedSum[11] +
               sharedSum[12] + sharedSum[13] + sharedSum[14] + sharedSum[15] +
               sharedSum[16] + sharedSum[17] + sharedSum[18] + sharedSum[19] +
               sharedSum[20] + sharedSum[21] + sharedSum[22] + sharedSum[23] +
               sharedSum[24] + sharedSum[25] + sharedSum[26] + sharedSum[27] +
               sharedSum[28] + sharedSum[29] + sharedSum[30] + sharedSum[31]);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int i = minI; i < maxI; ++i) {
        output[i] /= sum;
    }
}

/**
 for 2D the input/output data dimesion is (height, width)
 shape.dim0 is height
 shape.dim1 is widthX4
 shape.dim2 is width
 
 a thread will handle with a dim
 */
kernel void BROU(SoftMax2D)(device real *input           [[buffer(0)]],
                            device real *output          [[buffer(1)]],
                            constant TensorShape& shape  [[buffer(2)]],
                            ushort index [[thread_position_in_grid]]) {
    int height  = shape.dim0;
    int widthX4 = shape.dim1;
    int width   = shape.dim2;
    
    if (index >= height) {
        return;
    }
    
    device real *inputPtr  = input  + index * widthX4;
    device real *outputPtr = output + index * widthX4;
    
    real sum = 0;
    for (int i = 0; i < width; ++i) {
        real e = exp(inputPtr[i]);
        outputPtr[i] = e;
        sum += e;
    }
    
    for (int i = 0; i < width; ++i) {
        outputPtr[i] /= sum;
    }
}

/**
 for 3D input/output data the dimension is (height, width, channel)
 the memory's dimension is (height, width, channelX4)
 
 shape.dim0 is height
 shape.dim1 is width
 shape.dim2 is channelX4
 shape.dim3 is channel
 
 a thread will handle with a dim
 */
kernel void BROU(SoftMax3D)(device real *input           [[buffer(0)]],
                            device real *output          [[buffer(1)]],
                            constant TensorShape& shape  [[buffer(2)]],
                            ushort2 grid [[thread_position_in_grid]]) {
    int height    = shape.dim0;
    int width     = shape.dim1;
    int channelX4 = shape.dim2;
    int channel   = shape.dim3;
    
    int x = grid.x;
    int y = grid.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    device real *inputPtr  = input  + (y * width + x) * channelX4;
    device real *outputPtr = output + (y * width + x) * channelX4;
    
    real sum = 0;
    for (int i = 0; i < channel; ++i) {
        real e = exp(inputPtr[i]);
        outputPtr[i] = e;
        sum += e;
    }
    
    for (int i = 0; i < width; ++i) {
        outputPtr[i] /= sum;
    }
}

#endif



