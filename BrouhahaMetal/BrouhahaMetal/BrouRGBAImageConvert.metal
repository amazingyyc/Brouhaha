/**
 * BrouRGBAImageConvertLayer
 * Created by yanyuanchi on 2017/7/25.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * this layer just convert the uint8 RGBA image to half image
 */

#include <metal_stdlib>
#include <metal_math>

#include "BrouStruct.metal"

using namespace metal;

/**
 * convert the uint8_t image pixles data to half
 */
kernel void brouConvertRGBAUInt8ToHalf(device uchar *in  [[buffer(0)]],
                                       device half  *out [[buffer(1)]],
                                       constant TensorShape& shape [[buffer(2)]],
                                       ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;

    int x = grid.x << 2;
    int y = grid.y << 2;

    if (x >= width || y >= height) {
        return;
    }

    int minJ = y;
    int minI = x;
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);

    for (int j = minJ; j < maxJ; ++j) {
        for (int i = minI; i < maxI; ++i) {
            device uchar4 *inV  = (device uchar4*)(in + (j * width + i) * 4);
            device half4  *outV = (device half4*)(out + (j * width + i) * 4);
            
            outV[0] = static_cast<half4>(inV[0]);
        }
    }
}

kernel void brouConvertRGBAHalfToUInt8(device half  *in  [[buffer(0)]],
                                       device uchar *out [[buffer(1)]],
                                       constant TensorShape& shape [[buffer(2)]],
                                       ushort2 grid [[thread_position_in_grid]]) {
    int height = shape.dim0;
    int width  = shape.dim1;
    
    int x = grid.x << 2;
    int y = grid.y << 2;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int minJ = y;
    int minI = x;
    
    int maxJ = min(y + 4, height);
    int maxI = min(x + 4, width);
    
    for (int j = minJ; j < maxJ; ++j) {
        for (int i = minI; i < maxI; ++i) {
            device half4  *inV  = (device half4* )(in +  (j * width + i) * 4);
            device uchar4 *outV = (device uchar4*)(out + (j * width + i) * 4);
            
            outV[0] = static_cast<uchar4>(inV[0]);
        }
    }
}























