// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>  // for at::cuda::getCurrentCUDAStream

#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// new_xyz: (B, M, 3)
// xyz: (B, N, 3)
// idx: (B, M, nsample)
void ball_query_kernel_launcher(int b, int n, int m,
                                float min_radius, float max_radius,
                                int nsample,
                                const float *new_xyz,
                                const float *xyz,
                                int *idx,
                                cudaStream_t stream);

int ball_query_wrapper(int b, int n, int m,
                       float min_radius, float max_radius, int nsample,
                       at::Tensor new_xyz_tensor,
                       at::Tensor xyz_tensor,
                       at::Tensor idx_tensor) {
  CHECK_INPUT(new_xyz_tensor);
  CHECK_INPUT(xyz_tensor);

  const float *new_xyz = new_xyz_tensor.data_ptr<float>();
  const float *xyz = xyz_tensor.data_ptr<float>();
  int *idx = idx_tensor.data_ptr<int>();

  // PyTorch 2.x / CUDA 11.8 推荐写法
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  ball_query_kernel_launcher(b, n, m,
                             min_radius, max_radius,
                             nsample,
                             new_xyz,  // 注意顺序：new_xyz 在前
                             xyz,
                             idx,
                             stream);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_query_wrapper", &ball_query_wrapper, "ball_query_wrapper");
}
