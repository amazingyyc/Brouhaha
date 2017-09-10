#if defined(real) && defined(real4) && defined(BROU)

kernel void BROU(AddBias1D)(device real *in                [[buffer(0)]],
                            device real *bia               [[buffer(1)]],
                            device real *out               [[buffer(2)]],
                            constant TensorShape& shape    [[buffer(3)]],
                            ushort grid [[thread_position_in_grid]]) {
    int len = shape.dim0;
    int index = grid << 2;
    
    if (index >= len) {
        return;
    }
    
    device real4 *inV  = (device real4*)(in  + index);
    device real4 *biaV = (device real4*)(bia + index);
    device real4 *outV = (device real4*)(out + index);
    
    outV[0] = inV[0] + biaV[0];
}


/**
 * the in and out's dimension is (height, width)
 * the bias's dimension is (width, 1)
 * width is timed by 4
 */
kernel void BROU(AddBias2D)(device real *in                [[buffer(0)]],
                            device real *bia               [[buffer(1)]],
                            device real *out               [[buffer(2)]],
                            constant TensorShape& shape    [[buffer(3)]],
                            ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int y = grid.y << 2;
    int x = grid.x << 2;
    
    if (y >= height || x >= width) {
        return;
    }
    
    real4 biaV = ((device real4*)(bia + x))[0];
    
    int maxJ = min(y + 4, height);
    
    for (int j = y; j < maxJ; ++j) {
        int offset = j * width + x;
        
        device real4 *inV  = (device real4*)(in  + offset);
        device real4 *outV = (device real4*)(out + offset);
        
        outV[0] = inV[0] + biaV;
    }
}

/**
 * the in and out's dimension is (height, width, channel)
 * the bia diemnsion is (channel, 1)
 * the channel is timed by 4
 */
kernel void BROU(AddBias3D)(device real *in                [[buffer(0)]],
                            device real *bia               [[buffer(1)]],
                            device real *out               [[buffer(2)]],
                            constant TensorShape& shape    [[buffer(3)]],
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
    
    real4 biaV = ((device real4*)(bia + z))[0];
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            int offset = (j * width + i) * channel + z;
            
            device real4 *inV  = (device real4*)(in  + offset);
            device real4 *outV = (device real4*)(out + offset);
            
            outV[0] = inV[0] + biaV;
        }
    }
}

#endif


















