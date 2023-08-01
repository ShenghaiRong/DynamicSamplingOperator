#include <torch/extension.h>

#include <cmath>
#include <vector>

void ads_im2col(const at::Tensor data_im, const at::Tensor data_dilation,
                const at::Tensor stride_h, const at::Tensor stride_w,
                const int channels, const int height, const int width,
                const int ksize_h, const int ksize_w,
                const int pad_h, const int pad_w,
                const int parallel_imgs, const int ads_group,
                at::Tensor data_col);

void ads_col2im(const at::Tensor data_col, const at::Tensor data_dilation,
                const at::Tensor stride_h, const at::Tensor stride_w,
                const int channels, const int inputHeight, const int inputWidth,
                const int ksize_h, const int ksize_w,
                const int pad_h, const int pad_w,
                const int parallel_imgs, const int ads_group,
                at::Tensor grad_im);

void ads_col2im_coord(
    const at::Tensor data_col, const at::Tensor data_im,
    const at::Tensor data_dilation,
    const at::Tensor stride_h, const at::Tensor stride_w,
    const int channels, const int inputHeight,
    const int inputWidth, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int parallel_imgs, const int ads_group, at::Tensor grad_dilation);

void shape_check(at::Tensor input, at::Tensor dilation, at::Tensor *gradOutput,
                 at::Tensor stride_h, at::Tensor stride_w,
                 at::Tensor weight, int kH, int kW, int padH,
                 int padW, int group, int ads_group)
{
    TORCH_CHECK(weight.ndimension() == 4,
             "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
             "but got: %s",
             weight.ndimension());

    TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

    TORCH_CHECK(kW > 0 && kH > 0,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH,
             kW);

    TORCH_CHECK((weight.size(2) == kH && weight.size(3) == kW),
             "kernel size should be consistent with weight, ",
             "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
             kW, weight.size(2), weight.size(3));

    //TORCH_CHECK(dW > 0 && dH > 0,
    //         "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

    //TORCH_CHECK(
    //    dilationW > 0 && dilationH > 0,
    //    "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
    //    dilationH, dilationW);

    int ndim = input.ndimension();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4)
    {
        dimf++;
        dimh++;
        dimw++;
    }

    TORCH_CHECK(ndim == 3 || ndim == 4, "3D or 4D input tensor expected but got: %s",
             ndim);

    long nInputPlane = weight.size(1) * group;
    long inputHeight = input.size(dimh);
    long inputWidth = input.size(dimw);
    long nOutputPlane = weight.size(0);
    long outputHeight = stride_h.size(1);
    //  (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
    long outputWidth = stride_w.size(1);
    //  (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

    TORCH_CHECK(nInputPlane % ads_group == 0,
             "input channels must divide ads group size");

    if (outputWidth < 1 || outputHeight < 1)
        AT_ERROR(
            "Given input size: (%ld x %ld x %ld). "
            "Calculated output size: (%ld x %ld x %ld). Output size is too small",
            nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
            outputWidth);

    TORCH_CHECK(input.size(1) == nInputPlane,
             "invalid number of input planes, expected: %d, but got: %d",
             nInputPlane, input.size(1));

    TORCH_CHECK((inputHeight >= kH && inputWidth >= kW),
             "input image is smaller than kernel");

    TORCH_CHECK((dilation.size(2) == outputHeight && dilation.size(3) == outputWidth),
             "invalid spatial size of dilation, expected height: %d width: %d, but "
             "got height: %d width: %d",
             outputHeight, outputWidth, dilation.size(2), dilation.size(3));

    TORCH_CHECK((inputHeight >= outputHeight && inputWidth >= outputWidth),
             "input spatial size should be greater than ouput,"
             "input height: %d width: %d, but "
             "got output height: %d width: %d",
             inputHeight, inputWidth, outputHeight, outputWidth);

    TORCH_CHECK((dilation.size(1) == ads_group * 1),
             "invalid number of channels of offset");

    if (gradOutput != NULL)
    {
        TORCH_CHECK(gradOutput->size(dimf) == nOutputPlane,
                 "invalid number of gradOutput planes, expected: %d, but got: %d",
                 nOutputPlane, gradOutput->size(dimf));

        TORCH_CHECK((gradOutput->size(dimh) == outputHeight &&
                  gradOutput->size(dimw) == outputWidth),
                 "invalid size of gradOutput, expected height: %d width: %d , but "
                 "got height: %d width: %d",
                 outputHeight, outputWidth, gradOutput->size(dimh),
                 gradOutput->size(dimw));
    }
}

int ads_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                          at::Tensor dilation, at::Tensor output,
                          at::Tensor stride_h, at::Tensor stride_w,
                          at::Tensor columns, at::Tensor ones, int kW,
                          int kH, int padW, int padH,
                          int group, int ads_group, int im2col_step)
{

    shape_check(input, dilation, NULL, stride_h, stride_w,
                weight, kH, kW, padH, padW,
                group, ads_group);

    input = input.contiguous();
    dilation = dilation.contiguous();
    weight = weight.contiguous();
    stride_h = stride_h.contiguous();
    stride_w = stride_w.contiguous();

    int batch = 1;
    if (input.ndimension() == 3)
    {
        batch = 0;
        input.unsqueeze_(0);
        dilation.unsqueeze_(0);
        stride_h.unsqueeze_(0);
        stride_w.unsqueeze_(0);
    }

    long batchsize = input.size(0);
    long nInputPlane = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long nOutputPlane = weight.size(0);

    long outputWidth = stride_w.size(1);
    long outputHeight = stride_h.size(1);

    TORCH_CHECK((dilation.size(0) == batchsize), "invalid batch size of dilation");

    output = output.reshape({batchsize / im2col_step, im2col_step, nOutputPlane,
                          outputHeight, outputWidth});

    columns = at::zeros(
        {nInputPlane * kH * kW, im2col_step * outputHeight * outputWidth},
        input.options());

    if (ones.ndimension() != 2 ||
        ones.size(0) * ones.size(1) < outputHeight * outputWidth)
    {
        ones = at::ones({outputHeight, outputWidth}, input.options());
    }

    input = input.reshape({batchsize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth});

    dilation = dilation.reshape({batchsize / im2col_step, im2col_step,
                              ads_group, outputHeight, outputWidth});

    stride_h = stride_h.reshape({batchsize / im2col_step, im2col_step,
                              outputHeight, 1});

    stride_w = stride_w.reshape({batchsize / im2col_step, im2col_step,
                              outputWidth, 1});

    at::Tensor output_buffer =
        at::zeros({batchsize / im2col_step, nOutputPlane,
                   im2col_step * outputHeight, outputWidth},
                  output.options());

    output_buffer = output_buffer.reshape(
        {output_buffer.size(0), group, output_buffer.size(1) / group,
         output_buffer.size(2), output_buffer.size(3)});

    for (int elt = 0; elt < batchsize / im2col_step; elt++)
    {
        ads_im2col(input[elt], dilation[elt], stride_h[elt], stride_w[elt],
                   nInputPlane, inputHeight, inputWidth,
                   kH, kW, padH, padW, im2col_step,
                   ads_group, columns);

        columns = columns.reshape({group, columns.size(0) / group, columns.size(1)});

        weight = weight.reshape({group, weight.size(0) / group, weight.size(1),
                              weight.size(2), weight.size(3)});

        for (int g = 0; g < group; g++)
        {
            output_buffer[elt][g] = output_buffer[elt][g]
                                        .flatten(1)
                                        .addmm_(weight[g].flatten(1), columns[g])
                                        .reshape_as(output_buffer[elt][g]);
        }
    }

    output_buffer = output_buffer.reshape(
        {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
         output_buffer.size(3), output_buffer.size(4)});

    output_buffer = output_buffer.reshape({batchsize / im2col_step, nOutputPlane,
                                        im2col_step, outputHeight, outputWidth});

    output_buffer.transpose_(1, 2);
    output.copy_(output_buffer);
    output = output.reshape({batchsize, nOutputPlane, outputHeight, outputWidth});

    input = input.reshape({batchsize, nInputPlane, inputHeight, inputWidth});

    dilation = dilation.reshape(
        {batchsize, ads_group, outputHeight, outputWidth});

    stride_h = stride_h.reshape({batchsize, outputHeight, 1});
    stride_w = stride_w.reshape({batchsize, outputWidth, 1});

    if (batch == 0)
    {
        output = output.reshape({nOutputPlane, outputHeight, outputWidth});
        input = input.reshape({nInputPlane, inputHeight, inputWidth});
        dilation = dilation.reshape({ads_group, outputHeight, outputWidth});
        stride_h = stride_h.reshape({outputHeight, 1});
        stride_w = stride_h.reshape({outputWidth, 1});
    }

    return 1;
}

int ads_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor dilation, at::Tensor gradOutput,
    at::Tensor gradWeight, // at::Tensor gradBias,
    at::Tensor stride_h, at::Tensor stride_w,
    at::Tensor columns, at::Tensor ones, int kW, int kH,
    int padW, int padH, int group, int ads_group, float scale, int im2col_step)
{
    shape_check(input, dilation, &gradOutput, stride_h, stride_w,
                gradWeight, kH, kW, padH,
                padW, group, ads_group);

    input = input.contiguous();
    dilation = dilation.contiguous();
    gradOutput = gradOutput.contiguous();
    stride_h = stride_h.contiguous();
    stride_w = stride_w.contiguous();

    int batch = 1;

    if (input.ndimension() == 3)
    {
        batch = 0;
        input = input.reshape(
            at::IntList({1, input.size(0), input.size(1), input.size(2)}));
        gradOutput = gradOutput.reshape(
            {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
    }

    long batchsize = input.size(0);
    long nInputPlane = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long nOutputPlane = gradWeight.size(0);

    long outputHeight = stride_h.size(1);
    long outputWidth = stride_w.size(1);

    TORCH_CHECK((dilation.size(0) == batchsize), "invalid batch size of dilation");

    columns = at::zeros(
        {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
        input.options());

    gradOutput = gradOutput.reshape({batchsize / im2col_step, im2col_step,
                                  nOutputPlane, outputHeight, outputWidth});
    gradOutput.transpose_(1, 2);

    at::Tensor gradOutputBuffer = at::zeros_like(gradOutput);
    gradOutputBuffer =
        gradOutputBuffer.reshape({batchsize / im2col_step, nOutputPlane, im2col_step,
                               outputHeight, outputWidth});
    gradOutputBuffer.copy_(gradOutput);
    gradOutputBuffer =
        gradOutputBuffer.reshape({batchsize / im2col_step, nOutputPlane,
                               im2col_step * outputHeight, outputWidth});
    gradOutput.transpose_(1, 2);
    gradOutput =
        gradOutput.reshape({batchsize, nOutputPlane, outputHeight, outputWidth});

    input = input.reshape({batchsize / im2col_step, im2col_step, nInputPlane,
                        inputHeight, inputWidth});
    dilation = dilation.reshape({batchsize / im2col_step, im2col_step,
                              ads_group * 1, outputHeight, outputWidth});
    stride_h = stride_h.reshape({batchsize / im2col_step, im2col_step, outputHeight, 1});
    stride_w = stride_w.reshape({batchsize / im2col_step, im2col_step, outputWidth, 1});

    for (int elt = 0; elt < batchsize / im2col_step; elt++)
    {
        ads_im2col(input[elt], dilation[elt], stride_h[elt], stride_w[elt],
                   nInputPlane, inputHeight, inputWidth,
                   kH, kW, padH, padW, im2col_step,
                   ads_group, columns);

        gradOutputBuffer = gradOutputBuffer.reshape(
            {gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group,
             gradOutputBuffer.size(2), gradOutputBuffer.size(3)});
        columns = columns.reshape({group, columns.size(0) / group, columns.size(1)});
        gradWeight =
            gradWeight.reshape({group, gradWeight.size(0) / group, gradWeight.size(1),
                             gradWeight.size(2), gradWeight.size(3)});

        for (int g = 0; g < group; g++)
        {
            gradWeight[g] = gradWeight[g]
                                .flatten(1)
                                .addmm_(gradOutputBuffer[elt][g].flatten(1),
                                        columns[g].transpose_(1, 0), 1.0, scale)
                                .reshape_as(gradWeight[g]);
        }
        gradOutputBuffer = gradOutputBuffer.reshape(
            {gradOutputBuffer.size(0),
             gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
             gradOutputBuffer.size(3), gradOutputBuffer.size(4)});
        columns = columns.reshape({columns.size(0) * columns.size(1), columns.size(2)});
        gradWeight = gradWeight.reshape({gradWeight.size(0) * gradWeight.size(1),
                                      gradWeight.size(2), gradWeight.size(3),
                                      gradWeight.size(4)});
    }

    input = input.reshape({batchsize, nInputPlane, inputHeight, inputWidth});
    dilation = dilation.reshape({batchsize, ads_group, outputHeight, outputHeight});

    if (batch == 0)
    {
        gradOutput = gradOutput.reshape({nOutputPlane, outputHeight, outputWidth});
        input = input.reshape({nInputPlane, inputHeight, inputWidth});
    }

    return 1;
}

int ads_conv_backward_input_cuda(at::Tensor input, at::Tensor dilation,
                                 at::Tensor stride_h, at::Tensor stride_w,
                                 at::Tensor gradOutput, at::Tensor gradInput,
                                 at::Tensor gradDilation, at::Tensor weight,
                                 at::Tensor columns, int kW, int kH,
                                 int padW, int padH, int group,
                                 int ads_group, int im2col_step)
{
    shape_check(input, dilation, &gradOutput, stride_h, stride_w,
                weight, kH, kW, padH,
                padW, group, ads_group);

    input = input.contiguous();
    dilation = dilation.contiguous();
    gradOutput = gradOutput.contiguous();
    weight = weight.contiguous();

    int batch = 1;

    if (input.ndimension() == 3)
    {
        batch = 0;
        input = input.reshape({1, input.size(0), input.size(1), input.size(2)});
        dilation = dilation.reshape({1, dilation.size(0), dilation.size(1), dilation.size(2)});
        gradOutput = gradOutput.reshape(
            {1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
    }

    long batchsize = input.size(0);
    long nInputPlane = input.size(1);
    long inputHeight = input.size(2);
    long inputWidth = input.size(3);

    long nOutputPlane = weight.size(0);
    long outputHeight = stride_h.size(1);
    long outputWidth = stride_w.size(1);

    TORCH_CHECK((dilation.size(0) == batchsize), 3, "invalid batch size of dilation");
    gradInput = gradInput.reshape({batchsize, nInputPlane, inputHeight, inputWidth});
    columns = at::zeros(
        {nInputPlane * kH * kW, im2col_step * outputHeight * outputWidth},
        input.options());

    gradOutput = gradOutput.reshape({batchsize / im2col_step, im2col_step,
                                  nOutputPlane, outputHeight, outputWidth});
    gradOutput.transpose_(1, 2);

    gradInput = gradInput.reshape({batchsize / im2col_step, im2col_step,
                                nInputPlane, inputHeight, inputWidth});
    input = input.reshape({batchsize / im2col_step, im2col_step,
                        nInputPlane, inputHeight, inputWidth});

    gradDilation = gradDilation.reshape({batchsize / im2col_step, im2col_step,
                                      ads_group, outputHeight, outputWidth});
    dilation = dilation.reshape({batchsize / im2col_step, im2col_step,
                              ads_group * 1, outputHeight, outputWidth});
    stride_h = stride_h.reshape({batchsize / im2col_step, im2col_step, outputHeight, 1});
    stride_w = stride_w.reshape({batchsize / im2col_step, im2col_step, outputWidth, 1});

    for (int elt = 0; elt < batchsize / im2col_step; elt++)
    {
        columns = columns.reshape({group, columns.size(0) / group, columns.size(1)});
        weight = weight.reshape({group, weight.size(0) / group, weight.size(1),
                              weight.size(2), weight.size(3)});
        gradOutput = gradOutput.reshape({gradOutput.size(0), group,
                                      gradOutput.size(1) / group, gradOutput.size(2),
                                      gradOutput.size(3), gradOutput.size(4)});

        for (int g = 0; g < group; g++)
        {
            columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose_(0, 1),
                                           gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
        }

        columns = columns.reshape({columns.size(0) * columns.size(1), columns.size(2)});
        gradOutput = gradOutput.reshape(
            {gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2),
             gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)});

        ads_col2im(columns, dilation[elt], stride_h[elt], stride_w[elt],
                   nInputPlane, inputHeight, inputWidth,
                   kH, kW, padH, padW, im2col_step,
                   ads_group, gradInput[elt]);

        ads_col2im_coord(columns, input[elt], dilation[elt],
                         stride_h[elt], stride_w[elt], nInputPlane,
                         inputHeight, inputWidth, kH, kW, padH, padW,
                         im2col_step, ads_group, gradDilation[elt]);
    }

    gradOutput.transpose_(1, 2);
    gradOutput =
        gradOutput.reshape({batchsize, nOutputPlane, outputHeight, outputWidth});

    gradInput = gradInput.reshape({batchsize, nInputPlane, inputHeight, inputWidth});
    input = input.reshape({batchsize, nInputPlane, inputHeight, inputWidth});
    gradDilation = gradDilation.reshape(
        {batchsize, ads_group * 1, outputHeight, outputWidth});
    dilation = dilation.reshape(
        {batchsize, ads_group * 1, outputHeight, outputWidth});

    if (batch == 0)
    {
        gradOutput = gradOutput.reshape({nOutputPlane, outputHeight, outputWidth});
        gradInput = gradInput.reshape({nInputPlane, inputHeight, inputWidth});
        input = input.reshape({nInputPlane, inputHeight, inputWidth});
        gradDilation = gradDilation.reshape(
            {ads_group * 1, outputHeight, outputWidth});
        dilation = dilation.reshape({ads_group * 1, outputHeight, outputWidth});
    }

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ads_conv_forward_cuda", &ads_conv_forward_cuda,
          "ads forward (CUDA)");
    m.def("ads_conv_backward_input_cuda", &ads_conv_backward_input_cuda,
          "ads_conv_backward_input (CUDA)");
    m.def("ads_conv_backward_parameters_cuda",
          &ads_conv_backward_parameters_cuda,
          "ads_conv_backward_parameters (CUDA)");
}