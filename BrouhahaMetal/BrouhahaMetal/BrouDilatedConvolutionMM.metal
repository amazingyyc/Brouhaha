/**
 * handle the dialted convolution
 */

#include <metal_stdlib>
using namespace metal;

/**the height width and channel of the input data, and the input date's dimension is (height, width, channel)*/
constant int inputHeight[[function_constant(0)]];
constant int inputWidth[[function_constant(1)]];
constant int inputChannel[[function_constant(2)]];

/**the output data and the diemsion is (height, width, channel)*/
constant int outputHeight[[function_constant(3)]];
constant int outputWidth[[function_constant(4)]];
constant int outputChannel[[function_constant(5)]];

/**the kernel data, the kernel's dimesion is (outputchannel, height, width, inputchannel)*/
constant int kernelHeight[[function_constant(6)]];
constant int kernelWidth[[function_constant(7)]];

/**the pad of the input*/
constant int padLeft[[function_constant(8)]];
constant int padTop[[function_constant(9)]];

/**the step of the kernel*/
constant int strideX[[function_constant(10)]];
constant int strideY[[function_constant(11)]];

/**
 * the dilate params of dilated-convolution
 */
constant int dilatedX[[function_constant(12)]];
constant int dilatedY[[function_constant(13)]];

/**
 * inputchannelx4 >= inputchannel and timed by 4
 * outputchannelx4 >= outputchannel and timed by 4
 */
constant int inputChannelX4[[function_constant(14)]];
constant int outputChannelX4[[function_constant(15)]];

inline half4 getHalf4FromInput(device half *data, int height, int width, int channel, int y, int x, int z) {
    if (0 > y || 0 > x || 0 > z || y >= height || x >= width || z >= channel) {
        return 0;
    }

    device half4 *dataV = (device half4*)(data + (y * width + x) * channel + z);

    return dataV[0];
}

/**
 * the input'd dimension is (inputHeight, intputWidth, intputChannelX4)
 * it will be convert to matrix that is (kernelHeight * kernelWidth * inputChannel, [outputHeight * outputWidth]X4)
 * the matrix will be col-major
 *
 * the convolutoin is not equal the real convolution in math
 * like input is (a, b, c) the kernel is (i, j, k)
 * the convolution in math is output = a*k + b*j + c*i
 * but in brouhaha the convolution will be output = a*i + b*j + c*k
 */
kernel void brouConvertInput2MatrixDilated(device half *input   [[buffer(0)]],
                                           device half *matrix  [[buffer(1)]],
                                           ushort grid [[thread_position_in_grid]]) {
    /**every thread deal with 4 matrix cols*/
    int col = grid << 2;

    if (col >= outputHeight * outputWidth) {
        return;
    }

    int inputX0 = (col % outputWidth) * strideX - padLeft;
    int inputY0 = (col / outputWidth) * strideY - padTop;
    int inputX1 = ((col + 1) % outputWidth) * strideX - padLeft;
    int inputY1 = ((col + 1) / outputWidth) * strideY - padTop;
    int inputX2 = ((col + 2) % outputWidth) * strideX - padLeft;
    int inputY2 = ((col + 2) / outputWidth) * strideY - padTop;
    int inputX3 = ((col + 3) % outputWidth) * strideX - padLeft;
    int inputY3 = ((col + 3) / outputWidth) * strideY - padTop;

    int matrixColX4 = (outputHeight * outputWidth + 3) / 4 * 4;
    device half4 *matrixV = (device half4*)(matrix + col);

    int limit     = inputChannel - 3;
    int lastCount = inputChannel % 4;

    half4 inputV0, inputV1, inputV2, inputV3;

    for (int y = 0; y < kernelHeight; ++y) {
        for (int x = 0; x < kernelWidth; ++x) {
            int k = 0;
            for (; k < limit; k += 4) {
                inputV0 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY0+y*dilatedY, inputX0+x*dilatedX, k);
                inputV1 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY1+y*dilatedY, inputX1+x*dilatedX, k);
                inputV2 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY2+y*dilatedY, inputX2+x*dilatedX, k);
                inputV3 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY3+y*dilatedY, inputX3+x*dilatedX, k);

                matrixV[0] = {inputV0.x, inputV1.x, inputV2.x, inputV3.x};
                matrixV    = (device half4*)((device half*)matrixV + matrixColX4);

                matrixV[0] = {inputV0.y, inputV1.y, inputV2.y, inputV3.y};
                matrixV    = (device half4*)((device half*)matrixV + matrixColX4);

                matrixV[0] = {inputV0.z, inputV1.z, inputV2.z, inputV3.z};
                matrixV    = (device half4*)((device half*)matrixV + matrixColX4);

                matrixV[0] = {inputV0.w, inputV1.w, inputV2.w, inputV3.w};
                matrixV    = (device half4*)((device half*)matrixV + matrixColX4);
            }

            if (lastCount > 0) {
                inputV0 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY0 + y*dilatedY, inputX0 + x*dilatedX, k);
                inputV1 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY1 + y*dilatedY, inputX1 + x*dilatedX, k);
                inputV2 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY2 + y*dilatedY, inputX2 + x*dilatedX, k);
                inputV3 = getHalf4FromInput(input, inputHeight, inputWidth, inputChannelX4, inputY3 + y*dilatedY, inputX3 + x*dilatedX, k);

                for (int i = 0; i < lastCount; ++i) {
                    matrixV[0] = {inputV0[i], inputV1[i], inputV2[i], inputV3[i]};
                    matrixV = (device half4*)((device half*)matrixV + matrixColX4);
                }
            }
        }
    }
}












