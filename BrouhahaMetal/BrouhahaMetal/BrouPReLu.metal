#if defined(real) && defined(real4) && defined(BROU)

kernel void BROU(PReLu1D)(device real *input            [[buffer(0)]],
                          device real *output           [[buffer(1)]],
                          device real *a                [[buffer(2)]],
                          constant TensorShape& shape   [[buffer(3)]],
                          ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;
    
    if (index >= shape.dim0) {
        return;
    }
    
    real realA = a[0];
    
    real4 in = ((device real4*)(input + index))[0];
    real4 out;
    
    out.x = (in.x > 0) ? (in.x) : (realA * in.x);
    out.y = (in.y > 0) ? (in.y) : (realA * in.y);
    out.z = (in.z > 0) ? (in.z) : (realA * in.z);
    out.w = (in.w > 0) ? (in.w) : (realA * in.w);
    
    device real4 *outputV = (device real4*)(output + index);
    
    outputV[0] = out;
}

kernel void BROU(PReLu2D)(device real *input           [[buffer(0)]],
                          device real *output          [[buffer(1)]],
                          device real *a               [[buffer(2)]],
                          constant TensorShape& shape  [[buffer(3)]],
                          ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    
    if (y >= height || x >= width) {
        return;
    }
    
    real realA = a[0];
    
    int maxJ = min(y + 4, height);
    
    for (int j = y; j < maxJ; ++j) {
        int offset = j * width + x;
        
        real4 in = ((device real4*)(input  + offset))[0];
        real4 out;
        
        out.x = (in.x > 0) ? (in.x) : (realA * in.x);
        out.y = (in.y > 0) ? (in.y) : (realA * in.y);
        out.z = (in.z > 0) ? (in.z) : (realA * in.z);
        out.w = (in.w > 0) ? (in.w) : (realA * in.w);
        
        device real4 *outputV = (device real4*)(output + offset);
        
        outputV[0] = out;
    }
}

/**
 * every thread output 4X4X4 block
 * the channel must be timed by 4
 */
kernel void BROU(PReLu3D)(device real *input            [[buffer(0)]],
                          device real *output           [[buffer(1)]],
                          device real *a                [[buffer(2)]],
                          constant TensorShape& shape   [[buffer(3)]],
                          ushort3 grid [[thread_position_in_grid]]) {
    int height  = shape.dim0;
    int width   = shape.dim1;
    int channel = shape.dim2;
    
    int y = grid.y << 2;
    int x = grid.x << 2;
    int z = grid.z << 2;
    
    if (y >= height || x >= width || z >= channel) {
        return;
    }
    
    real realA = a[0];
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            int offset = (j * width + i) * channel + z;
            
            real4 in = ((device real4*)(input  + offset))[0];
            real4 out;
            
            out.x = (in.x > 0) ? (in.x) : (realA * in.x);
            out.y = (in.y > 0) ? (in.y) : (realA * in.y);
            out.z = (in.z > 0) ? (in.z) : (realA * in.z);
            out.w = (in.w > 0) ? (in.w) : (realA * in.w);
            
            device real4 *outputV = (device real4*)(output + offset);
            
            outputV[0] = out;
        }
    }
}
#endif
