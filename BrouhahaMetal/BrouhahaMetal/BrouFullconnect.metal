/**
 * Brouhaha
 * convolution.metal
 * Created by yanyuanchi on 2017/5/15.
 * Copyright © 2017年 yanyuanchi. All rights reserved.
 *
 * the fullconnet operate
 */
#if defined(real) && defined(real4) && defined(BROU)

/**
 * every thread will deal with 4 output
 * the input's dimesnion is (inputChannel, 1)
 * the output's dimension is (outputChannel, 1)
 * the weigths's dimesnion is (outputChannel, inputChannel)
 * the bias's dimension is (outputChannel, 1)
 *
 * inputchannel and outputchannel time by 4
 */
kernel void BROU(Fullconnect)(device real *input             [[buffer(0)]],
                              device real *weights           [[buffer(1)]],
                              device real *bia               [[buffer(2)]],
                              device real *output            [[buffer(3)]],
                              constant TensorShape& shape    [[buffer(4)]],
                              ushort grid [[thread_position_in_grid]]) {
    int inputChannel  = shape.dim0;
    int outputChannel = shape.dim1;
    
    int index = grid << 2;
    
    if (index >= outputChannel) {
        return;
    }
    
    real4 out = 0;
    real4 in;
    
    device real4 *inputV = (device real4*)input;
    
    device real4 *offset0 = (device real4*)(weights + inputChannel * index);
    device real4 *offset1 = (device real4*)(((device real*)offset0) + inputChannel);
    device real4 *offset2 = (device real4*)(((device real*)offset1) + inputChannel);
    device real4 *offset3 = (device real4*)(((device real*)offset2) + inputChannel);
    
    int loop = inputChannel / 4;
    
    do {
        in = inputV[0];
        
        out.x += dot(in, offset0[0]);
        out.y += dot(in, offset1[0]);
        out.z += dot(in, offset2[0]);
        out.w += dot(in, offset3[0]);
        
        inputV = (device real4*)(((device real*)inputV) + 4);
        
        offset0 = (device real4*)(((device real*)offset0) + 4);
        offset1 = (device real4*)(((device real*)offset1) + 4);
        offset2 = (device real4*)(((device real*)offset2) + 4);
        offset3 = (device real4*)(((device real*)offset3) + 4);
    } while(--loop);
    
    device real4 *outputV = (device real4*)(output + index);
    device real4 *biaV    = (device real4*)(bia + index);
    
    outputV[0] = out + biaV[0];
}

kernel void BROU(FullconnectWithoutBias)(device real *input             [[buffer(0)]],
                                         device real *weights           [[buffer(1)]],
                                         device real *output            [[buffer(2)]],
                                         constant TensorShape& shape    [[buffer(3)]],
                                         ushort grid [[thread_position_in_grid]]) {
    int inputChannel  = shape.dim0;
    int outputChannel = shape.dim1;
    
    int index = grid << 2;
    
    if (index >= outputChannel) {
        return;
    }
    
    real4 out = 0;
    real4 in;
    
    device real4 *inputV = (device real4*)input;
    
    device real4 *offset0 = (device real4*)(weights + inputChannel * index);
    device real4 *offset1 = (device real4*)(((device real*)offset0) + inputChannel);
    device real4 *offset2 = (device real4*)(((device real*)offset1) + inputChannel);
    device real4 *offset3 = (device real4*)(((device real*)offset2) + inputChannel);
    
    int loop = inputChannel / 4;
    
    do {
        in = inputV[0];
        
        out.x += dot(in, offset0[0]);
        out.y += dot(in, offset0[0]);
        out.z += dot(in, offset0[0]);
        out.w += dot(in, offset0[0]);
        
        inputV = (device real4*)(((device real*)inputV) + 4);
        
        offset0 = (device real4*)(((device real*)offset0) + 4);
        offset1 = (device real4*)(((device real*)offset1) + 4);
        offset2 = (device real4*)(((device real*)offset2) + 4);
        offset3 = (device real4*)(((device real*)offset3) + 4);
    } while(--loop);
    
    device real4 *outputV = (device real4*)(output + index);
    
    outputV[0] = out;
}

#endif













