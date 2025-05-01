// bitnet_linear.cu

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void bitlinear_forward_kernel(
    const uint8_t* __restrict__ input_packed,
    const uint8_t* __restrict__ weight_packed,
    float* __restrict__ output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output feature

    if (row < batch_size && col < out_features) {
        int packed_len = (in_features + 7) / 8;
        int dot = 0;
        for (int i = 0; i < packed_len; ++i) {
            uint8_t a = input_packed[row * packed_len + i];
            uint8_t b = weight_packed[col * packed_len + i];
            uint8_t xnor = ~(a ^ b);
            dot += __popc(xnor);  // Count number of matching bits
        }
        // Convert to signed dot product in {-1, 1} domain
        float rescaled = 2.0f * dot - in_features;
        output[row * out_features + col] = rescaled;
    }
}

void bitlinear_forward(
    at::Tensor input_packed,
    at::Tensor weight_packed,
    at::Tensor output,
    int in_features
) {
    const int batch_size = input_packed.size(0);
    const int out_features = weight_packed.size(0);
    const int packed_len = (in_features + 7) / 8;

    dim3 block(16, 16);
    dim3 grid((out_features + 15) / 16, (batch_size + 15) / 16);

    bitlinear_forward_kernel<<<grid, block>>>(
        input_packed.data_ptr<uint8_t>(),
        weight_packed.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_forward", &bitlinear_forward, "1-bit Linear Forward (CUDA)");
}
