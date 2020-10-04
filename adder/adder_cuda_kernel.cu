#include <ATen/ATen.h>
#include <THC/THC.h>
#include <THCUNN/SharedMem.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#define MAX_BLOCKS 256
#define NUM_THREADS 256
#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define HARDTANH(x) ((x) < (-1.0)) ? (-1.0) : (((x) <= (1.0)) ? (x) : (1.0))
const int WARP_SIZE = 32;
// Crude benchmarks suggest 256 is better than 512 and 1024
// TODO: Autotune/use better heuristics, improve speed more.
const int MAX_BLOCK_SIZE = 256;

static int getGradParamsNumThreads(int batchSize)
{
    //warp per item in a batch, up to a maximum
    return std::min(batchSize * WARP_SIZE, MAX_BLOCK_SIZE);
}

int get_blocks(int n)
{
    // return MAX(1, MIN(MAX_BLOCKS, (n - NUM_THREADS + 1) / NUM_THREADS));
    return MIN(MAX_BLOCKS, (n - NUM_THREADS + 1) / NUM_THREADS) + 1;
}

template <typename scalar_t>
__global__ void adder_forward_kernel(
    const scalar_t const *input,
    const scalar_t const *weight,
    // const scalar_t const *bias,
    scalar_t *output,
    const int num_elem,
    const int out_channels,
    const int in_channels,
    const int IW, const int IH,
    const int OW, const int OH,
    const int KW, const int KH,
    const int SW, const int SH,
    const int PW, const int PH)
{
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < num_elem; index += gridDim.x * blockDim.x)
    {
        const int n = index / OW / OH / out_channels;
        const int m = index / OW / OH % out_channels;
        const int h = index / OW % OH;
        const int w = index % OW;

        const scalar_t *p_weight = weight + m * in_channels * KH * KW;
        // scalar_t value = bias[m];
        scalar_t value = 0;
        // #pragma unroll
        for (int cc = 0; cc < in_channels; cc++)
        {
            // #pragma unroll
            const int image_offset0 = (n * in_channels + cc) * IH * IW;
            for (int kh = 0; kh < KH; kh++)
            {
                // #pragma unroll
                for (int kw = 0; kw < KW; kw++)
                {
                    const int ih = h * SH - PH + kh;
                    const int iw = w * SW - PW + kw;

                    bool boundary_condition = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
                    if (boundary_condition)
                    {
                        // value += input[image_offset0 + ih * IW + iw] * (*p_weight);
                        value -= abs(input[image_offset0 + ih * IW + iw] - (*p_weight));
                    }
                    else // padded area
                    {
                        value -= abs(*p_weight);
                    }
                    p_weight++;
                }
            }
        }
        output[index] = value;
    }
}

template <typename scalar_t>
__global__ void adder_backward_grad_in_kernel(
    scalar_t *grad_out,
    scalar_t *input,
    scalar_t *weight,
    scalar_t *grad_in,
    const int num_elem,
    const int out_channels,
    const int in_channels,
    const int IW, const int IH,
    const int OW, const int OH,
    const int KW, const int KH,
    const int SW, const int SH,
    const int PW, const int PH)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_elem; index += gridDim.x * blockDim.x)
    {
        const int n = index / IW / IH / in_channels;
        const int c = index / IW / IH % in_channels;
        const int h = index / IW % IH;
        const int w = index % IW;

        scalar_t value = 0;
        for (int mm = 0; mm < out_channels; mm++)
        {
            const int grad_out_offset0 = (n * out_channels + mm) * OH * OW;
            scalar_t *p_weight = weight + (mm * in_channels + c) * KH * KW;
            for (int kh = 0; kh < KH; kh++)
            {
                for (int kw = 0; kw < KW; kw++)
                {
                    int oh = h + PH - kh;
                    int ow = w + PW - kw;

                    if ((oh % SH == 0) && (ow % SW == 0))
                    {
                        const bool boundary_condition = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
                        if (boundary_condition)
                        {
                            oh = oh / SH;
                            ow = ow / SW;
                            // value += grad_out[grad_out_offset0 + oh * OW + ow] * (*p_weight);
                            scalar_t ht = HARDTANH(*p_weight - input[index]);
                            value += grad_out[grad_out_offset0 + oh * OW + ow] * ht;
                        }
                    }
                    p_weight++;
                }
            }
        }
        grad_in[index] = value;
    }
}

template <typename scalar_t>
__global__ void adder_backward_grad_weight_kernel(
    scalar_t *grad_out,
    scalar_t *input,
    scalar_t *weight,
    scalar_t *grad_weight,
    const int batch_size,
    const int out_channels,
    const int in_channels,
    const int IW, const int IH,
    const int OW, const int OH,
    const int KW, const int KH,
    const int SW, const int SH,
    const int PW, const int PH)
{
    SharedMem<scalar_t> smem;
    int bidx = blockIdx.x;
    int kW = bidx % KW;
    int kH = bidx / KW % KH;
    int ch = bidx / KW / KH % in_channels;
    int mh = bidx / KW / KH / in_channels;

    scalar_t grad = 0;
    const int laneId = threadIdx.x % WARP_SIZE;
    const int batch = threadIdx.x / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const int imageElements = OW * OH;
    for (int batchIdx = batch; batchIdx < batch_size; batchIdx += nwarps)
    {
        // Warp-stride loop over elements in a batch item
        for (int idx = laneId; idx < imageElements; idx += WARP_SIZE)
        {
            // Need to calculate the following: batch position, and offset into the gradOutput
            // in height, and width. We can intuit the corresponding position in the input from
            // the other parameters we have
            int go_w_offset = idx % OW;
            int go_h_offset = (idx / OW);

            int i_w_offset = go_w_offset * SW + kW - PW;
            int i_h_offset = go_h_offset * SH + kH - PH;

            int outputOffset = ((batchIdx * out_channels + mh) * OH) * OW + idx;
            if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < IW && i_h_offset < IH)
            {
                int inputOffset = ((batchIdx * in_channels + ch) * IH + i_h_offset) * IW + i_w_offset;
                // int outputOffset = ((batchIdx * out_channels + mh) * OH) * OW + idx;
                // grad += input[inputOffset] * grad_out[outputOffset];
                grad += (input[inputOffset] - weight[bidx]) * grad_out[outputOffset];
            }
            else // padded area
            {
                grad += - weight[bidx] * grad_out[outputOffset];
            }
        }
    }
    __syncthreads();
    scalar_t *buf = smem.getPointer();
    scalar_t tval = reduceBlock<scalar_t, ReduceAdd<scalar_t>>(
        buf, blockDim.x, grad, ReduceAdd<scalar_t>(), 0);

    // After reduction, first thread in the block has the gradient, so its responsible
    // for writing it to gradWeight
    if (threadIdx.x == 0)
    {
        int weightOffset = kW + (KW * kH) + (KW * KH * ch) + (KW * KH * in_channels * mh);
        grad_weight[weightOffset] = tval;
    }
}

////////////////////////////////////////////////////////////////////////
////////////////////////////END OF KERNEL///////////////////////////////
////////////////////////////////////////////////////////////////////////

int adder_cuda_forward(
    const at::Tensor &input,
    const at::Tensor &weight,
    // const at::Tensor &bias,
    at::Tensor &output,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    const int batch_size = output.size(0);
    const int in_channels = input.size(1);
    const int out_channels = output.size(1);
    const int IW = input.size(3);
    const int IH = input.size(2);
    const int OW = output.size(3);
    const int OH = output.size(2);
    const int num_elem = batch_size * out_channels * OH * OW;
    const int num_blocks = get_blocks(num_elem);

    AT_DISPATCH_FLOATING_TYPES(output.type(), "adder_cuda_forward", ([&] {
                                   adder_forward_kernel<scalar_t><<<num_blocks, NUM_THREADS>>>(
                                       input.data<scalar_t>(),
                                       weight.data<scalar_t>(),
                                       // bias.data<scalar_t>(),
                                       output.data<scalar_t>(),
                                       num_elem,
                                       out_channels,
                                       in_channels,
                                       IW, IH,
                                       OW, OH,
                                       KW, KH,
                                       SW, SH,
                                       PW, PH);
                               }));
    THCudaCheck(cudaGetLastError());
    return 1;
}

/*
scalar_t *grad_out,
scalar_t *weight,
scalar_t *grad_in,
const int num_elem,
const int out_channels,
const int in_channels,
const int IW, const int IH,
const int OW, const int OH,
const int KW, const int KH,
const int SW, const int SH,
const int PW, const int PH
*/

int adder_cuda_backward_grad_in(
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_in,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    const int batch_size = grad_in.size(0);
    const int in_channels = grad_in.size(1);
    const int out_channels = grad_out.size(1);
    const int IW = grad_in.size(3);
    const int IH = grad_in.size(2);
    const int OW = grad_out.size(3);
    const int OH = grad_out.size(2);
    const int num_elem = batch_size * in_channels * IH * IW;
    const int num_blocks = get_blocks(num_elem);

    AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "adder_cuda_backward_grad_in", ([&] {
                                   adder_backward_grad_in_kernel<scalar_t><<<num_blocks, NUM_THREADS>>>(
                                       grad_out.data<scalar_t>(),
                                       input.data<scalar_t>(),
                                       weight.data<scalar_t>(),
                                       grad_in.data<scalar_t>(),
                                       num_elem,
                                       out_channels,
                                       in_channels,
                                       IW, IH,
                                       OW, OH,
                                       KW, KH,
                                       SW, SH,
                                       PW, PH);
                               }));
    THCudaCheck(cudaGetLastError());
    return 1;
}

int adder_cuda_backward_grad_weight(
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_weight,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = grad_out.size(1);
    const int IW = input.size(3);
    const int IH = input.size(2);
    const int OW = grad_out.size(3);
    const int OH = grad_out.size(2);

    int blocks = out_channels * in_channels * KH * KW;
    // Make sure we have enough threads to perform the reduction, and use this number
    // to create the shared memory size for the reduction
    dim3 grid(blocks);
    dim3 block(getGradParamsNumThreads(batch_size));
    // int smem = block.x * sizeof(accreal);

    AT_DISPATCH_FLOATING_TYPES(grad_weight.type(), "adder_cuda_backward_grad_weight", ([&] {
                                   adder_backward_grad_weight_kernel<scalar_t><<<grid, block, block.x * sizeof(scalar_t)>>>(
                                       grad_out.data<scalar_t>(),
                                       input.data<scalar_t>(),
                                       weight.data<scalar_t>(),
                                       grad_weight.data<scalar_t>(),
                                       batch_size,
                                       out_channels,
                                       in_channels,
                                       IW, IH,
                                       OW, OH,
                                       KW, KH,
                                       SW, SH,
                                       PW, PH);
                               }));
    THCudaCheck(cudaGetLastError());
    return 1;
}

/*
scalar_t *grad_out,
scalar_t *input,
scalar_t *grad_weight,
const int batch_size,
const int out_channels,
const int in_channels,
const int IW, const int IH,
const int OW, const int OH,
const int KW, const int KH,
const int SW, const int SH,
const int PW, const int PH
*/
