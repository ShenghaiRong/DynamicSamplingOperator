 #include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <c10/macros/Macros.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

using namespace at;

#define CUDA_KERNEL_LOOP(i, n)  \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);  \
         i += blockDim.x * gridDim.x)


const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}


static __forceinline__ __device__
bool within_bounds_2d(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}


template <typename scalar_t>
__device__ scalar_t ads_im2col_bilinear(const scalar_t *bottom_data,
                                             const int data_width, 
                                             const int height, const int width,
                                             scalar_t h, scalar_t w)
{
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    scalar_t hh = 1 - lh;
    scalar_t hw = 1 - lw;

    scalar_t v1 = 0;
    if (within_bounds_2d(h_low, w_low, height, width))
        v1 = bottom_data[h_low * data_width + w_low];
    scalar_t v2 = 0;
    if (within_bounds_2d(h_low, w_high, height, width))
        v2 = bottom_data[h_low * data_width + w_high];
    scalar_t v3 = 0;
    if (within_bounds_2d(h_high, w_low, height, width))
        v3 = bottom_data[h_high *data_width + w_low];
    scalar_t v4 = 0;
    if (within_bounds_2d(h_high, w_high, height, width))
        v4 = bottom_data[h_high *data_width + w_high];
    
    scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    scalar_t val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    return val;
}


template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                        const int h, const int w, const int height, const int width)
{

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}


template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                            scalar_t dilation_h, scalar_t dilation_w,
                                            int i, int j,
                                            const int height, const int width, const scalar_t *im_data,
                                            const int data_width)
{

    if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
    {
        //empty
        return 0;
    }

    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;
    int dilation_h_low = floor(dilation_h * i);
    int dilation_w_low = floor(dilation_w * j);

    scalar_t weight = 0;


    if (argmax_h_low >= 0 && argmax_w_low >= 0)
    {
        scalar_t w1 = 2 * i * j * dilation_h - i * dilation_w_low - j * dilation_h_low - i - j;
        weight += w1 * im_data[argmax_h_low * data_width + argmax_w_low];
    }
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
    {
        scalar_t w2 = -2 * i * j * dilation_h + i * dilation_w_low + j * dilation_h_low + j;
        weight += w2 * im_data[argmax_h_low * data_width + argmax_w_high];
    }
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
    {
        scalar_t w3 = -2 * i * j * dilation_h + i * dilation_w_low + j * dilation_h_low + i;
        weight += w3 * im_data[argmax_h_high * data_width + argmax_w_low];
    }
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
    {
        scalar_t w4 = 2 * i * j * dilation_h - i * dilation_w_low - j * dilation_h_low;
        weight += w4 * im_data[argmax_h_high * data_width + argmax_w_high];
    }

    return weight;
}

template <typename scalar_t>
__global__ void ads_im2col_gpu_kernel(const int n, const scalar_t *data_im,
                                           const scalar_t *data_dilation,
                                           const scalar_t *stride_h,
                                           const scalar_t *stride_w,
                                           const int inputHeight, const int inputWidth, 
                                           const int kernel_h, const int kernel_w, 
                                           const int pad_h, const int pad_w,
                                           const int channel_per_ads_group,
                                           const int batch_size, 
                                           const int num_channels, 
                                           const int ads_group,
                                           const int height_col, const int width_col,
                                           scalar_t *data_col)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        // index idex of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / height_col / width_col) % batch_size;
        const int c_im = index / width_col / height_col / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;

        const int ads_group_index = c_im / channel_per_ads_group;

        scalar_t *data_col_ptr = \
            data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        
        const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * inputHeight * inputWidth;
        const scalar_t *data_dilation_ptr = \
            data_dilation + (b_col * ads_group + ads_group_index) * height_col * width_col;
        const scalar_t *stride_h_ptr = stride_h + (b_col * height_col);
        const scalar_t *stride_w_ptr = stride_w + (b_col * width_col);
        
        const int data_dilation_h_ptr = h_col * width_col + w_col;
        const int data_dilation_w_ptr = h_col * width_col + w_col;
        const scalar_t dilation_h = data_dilation_ptr[data_dilation_h_ptr];
        const scalar_t dilation_w = data_dilation_ptr[data_dilation_w_ptr];

        const scalar_t h_in = stride_h_ptr[h_col] - dilation_h;
        const scalar_t w_in = stride_w_ptr[w_col] - dilation_w;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                scalar_t val = static_cast<scalar_t>(0);
                const scalar_t h_im = h_in + i * dilation_h;
                const scalar_t w_im = w_in + j * dilation_w;
                if (h_im > -1 && w_im > -1 && h_im < inputHeight && w_im < inputWidth)
                {
                    val = ads_im2col_bilinear(data_im_ptr, inputWidth, inputHeight, inputWidth, h_im, w_im);
                }
                *data_col_ptr = val;
                data_col_ptr += batch_size * height_col * width_col;
            }
        }
    }
}


void ads_im2col(const at::Tensor data_im, const at::Tensor data_dilation,
    const at::Tensor stride_h, const at::Tensor stride_w,
    const int channels, const int inputHeight, const int inputWidth,
    const int ksize_h, const int ksize_w, 
    const int pad_h, const int pad_w,
    const int parallel_imgs, const int ads_group,
    at::Tensor data_col)
{
    int height_col = stride_h.size(1);
    int width_col = stride_w.size(1);
    int num_kernels = channels * height_col * width_col * parallel_imgs;
    int channel_per_ads_group = channels / ads_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_im.scalar_type(), "ads_im2col_gpu", ([&] {
            const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
            const scalar_t *data_dilation_ = data_dilation.data_ptr<scalar_t>();
            const scalar_t *stride_h_ = stride_h.data_ptr<scalar_t>();
            const scalar_t *stride_w_ = stride_w.data_ptr<scalar_t>(); 
            scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

            ads_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, data_im_, data_dilation_, stride_h_, stride_w_, 
                inputHeight, inputWidth, ksize_h, ksize_w,
                pad_h, pad_w, channel_per_ads_group, 
                parallel_imgs, channels, ads_group, height_col, width_col,
                data_col_);
            
        }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ads_im2col: %s\n", cudaGetErrorString(err));
    }

}




template <typename scalar_t>
__global__ void ads_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_dilation,
    const scalar_t *stride_h, const scalar_t *stride_w,
    const int channels, const int inputHeight, const int inputWidth, 
    const int ksize_h, const int ksize_w, const int pad_h, const int pad_w, 
    const int channel_per_ads_group, const int batch_size,
    const int ads_group, const int height_col, const int width_col,
    scalar_t *grad_im)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        const int j = (index / width_col / height_col / batch_size) % ksize_w;
        const int i = (index / width_col / height_col / batch_size / ksize_w) % ksize_h;
        const int c = index / width_col / height_col / batch_size / ksize_w / ksize_h;

        const int ads_group_index = c / channel_per_ads_group;

        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        int b = (index / width_col / height_col) % batch_size;
        
        const scalar_t *data_dilation_ptr = data_dilation + (b * ads_group + ads_group_index) *
                                                                1 * height_col * width_col;
        const scalar_t *stride_h_ptr = stride_h + (b * height_col);
        const scalar_t *stride_w_ptr = stride_w + (b * width_col);
        const int data_dilation_h_ptr = h_out * width_col + w_out;
        const int data_dilation_w_ptr = h_out * width_col + w_out; 
        scalar_t dilation_h = data_dilation_ptr[data_dilation_h_ptr];
        scalar_t dilation_w = data_dilation_ptr[data_dilation_w_ptr];
        scalar_t w_in = stride_w_ptr[w_out] - dilation_w;
        scalar_t h_in = stride_h_ptr[h_out] - dilation_h;
        const scalar_t cur_inv_h_data = h_in + i * dilation_h;
        const scalar_t cur_inv_w_data = w_in + j * dilation_w;

        const scalar_t cur_top_grad = data_col[index];
        const int cur_h = (int)cur_inv_h_data;
        const int cur_w = (int)cur_inv_w_data;

        for (int dy = -2; dy <= 2; dy++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                if (cur_h + dy >= 0 && cur_h + dy < inputHeight &&
                    cur_w + dx >= 0 && cur_w + dx < inputWidth &&
                    abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
                    abs(cur_inv_w_data - (cur_w + dx)) < 1)
                {
                    int cur_bottom_grad_pos = ((b * channels + c) * inputHeight + cur_h + dy) * inputWidth + cur_w + dx;
                    scalar_t weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, inputHeight, inputWidth);
                    atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }

    }
}


void ads_col2im(
    const at::Tensor data_col, const at::Tensor data_dilation, 
    const at::Tensor stride_h, const at::Tensor stride_w, 
    const int channels, const int inputHeight, const int inputWidth, 
    const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w,
    const int parallel_imgs, const int ads_group, at::Tensor grad_im)
{
    int height_col = stride_h.size(1);
    int width_col = stride_w.size(1);
    int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
    int channel_per_ads_group = channels / ads_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_col.scalar_type(), "ads_col2im_gpu", ([&] {
            const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
            const scalar_t *data_dilation_ = data_dilation.data_ptr<scalar_t>();
            const scalar_t *stride_h_ = stride_h.data_ptr<scalar_t>();
            const scalar_t *stride_w_ = stride_w.data_ptr<scalar_t>();
            scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

            ads_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
                num_kernels, data_col_, data_dilation_, stride_h_, stride_w_,
                channels, inputHeight, inputWidth,
                ksize_h, ksize_w, pad_h, pad_w,
                channel_per_ads_group, parallel_imgs,
                ads_group, height_col, width_col, grad_im_);
        }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in ads_col2im: %s\n", cudaGetErrorString(err));
    }
}





template <typename scalar_t>
__global__ void ads_col2im_coord_gpu_kernel(const int n, const scalar_t *data_col,
                                            const scalar_t *data_im, const scalar_t *data_dilation,
                                            const scalar_t *stride_h, const scalar_t *stride_w,
                                            const int channels, const int height, const int width,
                                            const int kernel_h, const int kernel_w,
                                            const int pad_h, const int pad_w,
                                            const int channel_per_ads_group,
                                            const int batch_size, const int dilation_channels, 
                                            const int ads_group, const int height_col,
                                            const int width_col, scalar_t *grad_dilation)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        scalar_t val = 0;
        int w = index % width_col;
        int h = (index / width_col) % height_col;
        int c = (index / width_col / height_col) % dilation_channels;
        int b = index / width_col / height_col / dilation_channels;

        const int ads_group_index = c;
        // const int col_step = 1;
        int cnt = 0;
        const scalar_t *data_col_ptr = data_col + ads_group_index * channel_per_ads_group *
                                                        batch_size * width_col * height_col;
        const scalar_t *data_im_ptr = data_im + (b * ads_group + ads_group_index) *
                                                    channel_per_ads_group / kernel_h / kernel_w * height * width;

        const scalar_t *data_dilation_ptr = data_dilation + (b * ads_group + ads_group_index) *
                                                                1 * height_col * width_col;
        const scalar_t *stride_h_ptr = stride_h + (b * height_col);
        const scalar_t *stride_w_ptr = stride_w + (b * width_col);

        // const int offset_c = c - ads_group_index * 1;                                                       
        
        for (int col_c = 0; col_c < channel_per_ads_group; col_c++)
        {
            const int col_pos = (((col_c * batch_size +b) * height_col) + h) * width_col + w;
            int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
            int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
            int w_out = col_pos % width_col;
            int h_out = (col_pos / width_col) % height_col;
            const int data_dilation_h_ptr = h_out * width_col + w_out;
            const int data_dilation_w_ptr = h_out * width_col + w_out;
            const scalar_t dilation_h = data_dilation_ptr[data_dilation_h_ptr];
            const scalar_t dilation_w = data_dilation_ptr[data_dilation_w_ptr];
            scalar_t h_in = stride_h_ptr[h_out] - dilation_h;
            scalar_t w_in = stride_w_ptr[w_out] - dilation_w;
            scalar_t inv_h = h_in + i * dilation_h;
            scalar_t inv_w = w_in + j * dilation_w;
            if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
            {
                inv_h = inv_w = -2;
            }
            const scalar_t weight = get_coordinate_weight(
                inv_h, inv_w, dilation_h, dilation_w, i, j,
                height, width, data_im_ptr + cnt * height * width, width);
            val += weight * data_col_ptr[col_pos];
            cnt = col_c / 9;
        }

        grad_dilation[index] = val;
    }
}





void ads_col2im_coord(
    const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_dilation,
    const at::Tensor stride_h, const at::Tensor stride_w,
    const int channels, const int inputHeight, const int inputWidth, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w,
    const int parallel_imgs, const int ads_group,
    at::Tensor grad_dilation)
{
    int height_col = stride_h.size(1);
    int width_col = stride_w.size(1);
    int num_kernels = parallel_imgs * ads_group * 1 * height_col * width_col;
    int channel_per_ads_group = channels * ksize_h * ksize_w / ads_group;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_col.scalar_type(), "ads_col2im_coord_gpu", ([&] {
          const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
          const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
          const scalar_t *data_dilation_ = data_dilation.data_ptr<scalar_t>();
          const scalar_t *stride_h_ = stride_h.data_ptr<scalar_t>();
          const scalar_t *stride_w_ = stride_w.data_ptr<scalar_t>();
          scalar_t *grad_dilation_ = grad_dilation.data_ptr<scalar_t>();
  
          ads_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
              num_kernels, data_col_, data_im_, data_dilation_, 
              stride_h_, stride_w_,
              channels, inputHeight, inputWidth,
              ksize_h, ksize_w, pad_h, pad_w,
              channel_per_ads_group,
              parallel_imgs, 1 * ads_group, ads_group,
              height_col, width_col, grad_dilation_);
        }));
}