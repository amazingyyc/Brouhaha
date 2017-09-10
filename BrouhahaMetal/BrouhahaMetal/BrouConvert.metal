#if defined(from) && defined(to) && defined(from4) && defined(to4) && defined(BROU_CONVERT)

kernel void BROU_CONVERT(from, to, 1D)(device from *input           [[buffer(0)]],
                                       device to   *output          [[buffer(1)]],
                                       constant TensorShape& shape  [[buffer(2)]],
                                       ushort grid [[thread_position_in_grid]]) {
    int index = grid << 2;
    
    if (index >= shape.dim0) {
        return;
    }
    
    device from4 *inputV  = (device from4*)(input  + index);
    device to4   *outputV = (device   to4*)(output + index);
    
    outputV[0] = static_cast<to4>(inputV[0]);
}

kernel void BROU_CONVERT(from, to, 2D)(device from *input           [[buffer(0)]],
                                       device to   *output          [[buffer(1)]],
                                       constant TensorShape& shape  [[buffer(2)]],
                                       ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int y = grid.y << 2;
    int x = grid.x << 2;
    
    if (y >= height || x >= width) {
        return;
    }
    
    int maxY = min(y + 4, height);
    
    for (int j = y; j < maxY; ++j) {
        int offset = j * width + x;
        
        device from4 *inputV  = (device from4*)(input  + offset);
        device to4   *outputV = (device   to4*)(output + offset);
        
        outputV[0] = static_cast<to4>(inputV[0]);
    }
}

kernel void BROU_CONVERT(from, to, 3D)(device from *input           [[buffer(0)]],
                                       device to   *output          [[buffer(1)]],
                                       constant TensorShape& shape  [[buffer(2)]],
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
    
    int maxY = min(y + 4, height);
    int maxX = min(x + 4, width);
    
    for (int j = y; j < maxY; ++j) {
        for (int i = x; i < maxX; ++i) {
            int offset = (j * width + i) * channel + z;
            
            device from4 *inputV  = (device from4*)(input  + offset);
            device to4   *outputV = (device   to4*)(output + offset);
            
            outputV[0] = static_cast<to4>(inputV[0]);
        }
    }
}

#endif









