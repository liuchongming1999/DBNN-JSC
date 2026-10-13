#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
// CUTLASS's FastF32 path combines several TF32 products for a closer FP32
// approximation. It is screened independently; it is not labelled exact FP32.
extern "C" int dbnn_cutlass_tensor(void* a,void* b,void* c,int M,int K,void* stream) {
  using Gemm=cutlass::gemm::device::Gemm<float,cutlass::layout::RowMajor,
    float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128,128,32>,cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<16,8,8>,
    cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,4,4,false,
    cutlass::arch::OpMultiplyAddFastF32>;
  typename Gemm::Arguments args({M,K,K},{(float*)a,K},{(float*)b,K},
    {(float*)c,K},{(float*)c,K},{1.0f,0.0f});
  Gemm gemm;auto status=gemm(args,nullptr,(cudaStream_t)stream);
  return status==cutlass::Status::kSuccess?0:1000+int(status);
}
