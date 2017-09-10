#if defined(real) && defined(real4) && defined(BROU)

kernel void BROU(Add1D)(device real *in1               [[buffer(0)]],
                        device real *in2               [[buffer(1)]],
                        device real *out               [[buffer(2)]],
                        constant TensorShape& shape    [[buffer(3)]],
                        ushort grid [[thread_position_in_grid]]) {
    int len = shape.dim0;
    
    int index = grid << 2;
    
    if (index >= len) {
        return;
    }
    
    device real4 *in1V = (device real4*)(in1 + index);
    device real4 *in2V = (device real4*)(in2 + index);
    device real4 *outV = (device real4*)(out + index);
    
    outV[0] = in1V[0] + in2V[0];
}

kernel void BROU(Add2D)(device real *in1               [[buffer(0)]],
                        device real *in2               [[buffer(1)]],
                        device real *out               [[buffer(2)]],
                        constant TensorShape& shape    [[buffer(3)]],
                        ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    
    if (y >= height || x >= width) {
        return;
    }
    
    int maxJ = min(y + 4, height);
    
    for (int j = y; j < maxJ; ++j) {
        int offset = j * width + x;
        
        device real4 *in1V = (device real4*)(in1 + offset);
        device real4 *in2V = (device real4*)(in2 + offset);
        device real4 *outV = (device real4*)(out + offset);
        
        outV[0] = in1V[0] + in2V[0];
    }
}

kernel void BROU(Add3D)(device real *in1               [[buffer(0)]],
                        device real *in2               [[buffer(1)]],
                        device real *out               [[buffer(2)]],
                        constant TensorShape& shape    [[buffer(3)]],
                        ushort3 grid [[thread_position_in_grid]]) {
    int height  = shape.dim0;
    int width   = shape.dim1;
    int channel = shape.dim2;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;
    
    if (y >= height || x >= width || z >= channel) {
        return;
    }
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            int offset = (j * width + i) * channel + z;
            
            device real4 *in1V = (device real4*)(in1 + offset);
            device real4 *in2V = (device real4*)(in2 + offset);
            device real4 *outV = (device real4*)(out + offset);
            
            outV[0] = in1V[0] + in2V[0];
        }
    }
}

#endif















