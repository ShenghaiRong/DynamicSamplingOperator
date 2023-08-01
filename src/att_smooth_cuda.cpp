#include <torch/extension.h>

#include <cmath>
#include <vector>

void attgridgen_gpu(at::Tensor attx, at::Tensor atty,
                    const int batch_size, const int att_size, const int out_size,
                    const float threshold, const int iters);


int forward(at::Tensor attx, at::Tensor atty,
            int input_size, int out_size, int dense, int iters, float scale)
{
    attx = attx.contiguous();
    atty = atty.contiguous();

    //shape_check(attx, atty, batch_size, att_size);
    long batch_size = attx.size(0);
    long att_size = attx.size(1);
    float threshold = scale * dense * input_size / att_size;

    //at::Tensor map_xi = at::ones({batch_size, att_size, 1}, attx.options());
    //at::Tensor map_yi = at::ones({batch_size, att_size, 1}, atty.options());
    //at::Tensor map_sum_x = at::ones({batch_size, 1, 1}, attx.options());
    //at::Tensor map_sum_y = at::ones({batch_size, 1, 1}, atty.options());

    //at::Tensor index_x = at::ones({batch_size, out_size, 1}, attx.options());
    //at::Tensor index_y = at::ones({batch_size, out_size, 1}, atty.options());

    attgridgen_gpu(attx, atty, batch_size, att_size, out_size, threshold, iters);


    //attx = attx.view({batch_size, input_size, 1});
    //atty = atty.view({batch_size, input_size, 1});

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &forward,
          "attgridgenerator forward (CUDA)");
}
