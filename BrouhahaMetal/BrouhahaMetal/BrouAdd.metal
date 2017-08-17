/**
 * add operate
 * Created by yanyuanchi on 2017/7/18.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the add operate is on n-dimensional array (n = 1, 2, 3)
 * support the Broadcasting rule, ref:https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
 */

#include <metal_stdlib>

#include "BrouStruct.metal"

using namespace metal;

/**
 * for compatible the other layers the addlayer have some constrain
 * for 1d 2d 3d tensor the diemnsion is (dim0) (dim0, dim1) (dim0, dim1, dim2)
 *
 * the last dim of in tensor must be 1 or time by 4, and can not be 1 at the same time
 * for in1(31, 1) in2(1, 1) can be changed to in1(31) in2(1)
 *
 * like 1d tensor the dimension of in1, in2, out is (in1-dim0) (in2-dim0) (out-dim0)
 * the in1-dim0 and in2-dim0 must be 1 or timed by 4 and in1-dim0 and in2-dim0 can not be 1 at same time
 */


inline half4 getHalf4From1D(device half *data, int len, int index) {
    if (1 == len) {
        return {data[0], data[0], data[0], data[0]};
    }

    device half4 *dataV = (device half4*)(data + index);

    return dataV[0];
}

/**
 * for 1d out every thread deal with 4 out
 * the out len must be timed by 4
 */
kernel void brouAdd1D(device half *in1           [[buffer(0)]],
                      device half *in2           [[buffer(1)]],
                      device half *out           [[buffer(2)]],
                      constant TensorShape& in1Shape [[buffer(3)]],
                      constant TensorShape& in2Shape [[buffer(4)]],
                      constant TensorShape& outShape [[buffer(5)]],
                      ushort grid [[thread_position_in_grid]]) {
    int in1Len = in1Shape.dim0;
    int in2Len = in2Shape.dim0;
    int outLen = outShape.dim0;
    
    int index = grid << 2;
    
    if (index >= outLen) {
        return;
    }
    
    half4 in1V = getHalf4From1D(in1, in1Len, index);
    half4 in2V = getHalf4From1D(in2, in2Len, index);
    
    device half4 *outV = (device half4*)(out + index);
    
    outV[0] = in1V + in2V;
}

inline half4 getHalf4From2D(device half *data, int height, int width, int y, int x) {
    if (1 == height) {
        y = 0;
    }
    
    if (1 == width) {
        device half *dataV = data + y;
        
        return {dataV[0], dataV[0], dataV[0], dataV[0]};
    } else {
        device half4 *dataV = (device half4*)(data + y * width + x);
        
        return dataV[0];
    }
}

/**
 * for 2d, every thread deal with 4 row
 * for a out (outheight, outwidth)
 * the outWidth must be timed by 4
 */
kernel void brouAdd2D(device half *in1           [[buffer(0)]],
                      device half *in2           [[buffer(1)]],
                      device half *out           [[buffer(2)]],
                      constant TensorShape& in1Shape [[buffer(3)]],
                      constant TensorShape& in2Shape [[buffer(4)]],
                      constant TensorShape& outShape [[buffer(5)]],
                      ushort grid [[thread_position_in_grid]]) {
    int outHeight = outShape.dim0;
    int outWidth  = outShape.dim1;
    
    int y = grid << 2;
    
    if (y >= outHeight) {
        return;
    }
    
    int in1Height = in1Shape.dim0;
    int in1Width  = in1Shape.dim1;
    
    int in2Height = in2Shape.dim0;
    int in2Width  = in2Shape.dim1;
    
    int maxJ = min(y + 4, outHeight);
    
    for (int j = y; j < maxJ; ++j) {
        for (int i = 0 ; i < outWidth; i += 4) {
            half4 in1V = getHalf4From2D(in1, in1Height, in1Width, j, i);
            half4 in2V = getHalf4From2D(in2, in2Height, in2Width, j, i);
            
            device half4 *outV = (device half4*)(out + j * outWidth + i);
            
            outV[0] = in1V + in2V;
        }
    }
}

inline half4 getHalf4From3D(device half *data, int height, int width, int channel, int y, int x, int z) {
    if (1 == height) {
        y = 0;
    }

    if (1 == width) {
        x = 0;
    }

    if (1 == channel) {
        device half *dataV = data + y * width + x;

        return {dataV[0], dataV[0], dataV[0], dataV[0]};
    } else {
        device half4 *dataV = (device half4*)(data + (y * width + x) * channel + z);

        return dataV[0];
    }
}

/**
 * for 3d every thread deal with 4X4X4 block
 * for a out (outheight, outwidth, outchannel)
 * the outchannel must be timed by 4
 */
kernel void brouAdd3D(device half *in1           [[buffer(0)]],
                      device half *in2           [[buffer(1)]],
                      device half *out           [[buffer(2)]],
                      constant TensorShape& in1Shape [[buffer(3)]],
                      constant TensorShape& in2Shape [[buffer(4)]],
                      constant TensorShape& outShape [[buffer(5)]],
                      ushort3 grid [[thread_position_in_grid]]) {
    int outHeight  = outShape.dim0;
    int outWidth   = outShape.dim1;
    int outChannel = outShape.dim2;

    int x = grid.x << 2;
    int y = grid.y << 2;
    int z = grid.z << 2;

    if (y >= outHeight || x >= outWidth || z >= outChannel) {
        return;
    }

    int in1Height  = in1Shape.dim0;
    int in1Width   = in1Shape.dim1;
    int in1Channel = in1Shape.dim2;

    int in2Height  = in2Shape.dim0;
    int in2Width   = in2Shape.dim1;
    int in2Channel = in2Shape.dim2;

    int maxJ = min(y + 4, outHeight);
    int maxI = min(x + 4, outWidth);

    for (int j = y; j < maxJ; ++j) {
        for (int i = x; i < maxI; ++i) {
            half4 in1V = getHalf4From3D(in1, in1Height, in1Width, in1Channel, j, i, z);
            half4 in2V = getHalf4From3D(in2, in2Height, in2Width, in2Channel, j, i, z);

            device half4 *outV = (device half4*)(out + (j * outWidth + i) * outChannel + z);

            outV[0] = in1V + in2V;
        }
    }
}













