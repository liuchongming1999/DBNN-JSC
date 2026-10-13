#include <cuda_runtime.h>
#include <cub/cub.cuh>

// Pure q(y)=y^T W y. Same FP32 arithmetic policy as native_kernels.cu.
__global__ void quadratic_reduce(const float* y,const float* z,float* q,int K) {
  using Reduce=cub::BlockReduce<float,128>;
  __shared__ typename Reduce::TempStorage tmp;
  int row=blockIdx.x;
  float sum=0;
  for(int k=threadIdx.x;k<K;k+=128) sum+=y[row*K+k]*z[row*K+k];
  sum=Reduce(tmp).Sum(sum);
  if(threadIdx.x==0) q[row]=sum;
}

__global__ void quadratic_direct(const float* y,const float* w,float* q,int K) {
  using Reduce=cub::BlockReduce<float,128>;
  __shared__ typename Reduce::TempStorage tmp;
  extern __shared__ float ys[];
  int row=blockIdx.x;
  for(int k=threadIdx.x;k<K;k+=128) ys[k]=y[row*K+k];
  __syncthreads();
  float sum=0;
  for(int i=threadIdx.x;i<K;i+=128) {
    float z=0;
    for(int j=0;j<K;j++) z+=ys[j]*w[j*K+i];
    sum+=ys[i]*z;
  }
  sum=Reduce(tmp).Sum(sum);
  if(threadIdx.x==0) q[row]=sum;
}

extern "C" int dbnn_q_reduce(void* y,void* z,void* q,int R,int K,void* stream) {
  quadratic_reduce<<<R,128,0,(cudaStream_t)stream>>>((float*)y,(float*)z,(float*)q,K);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_q_direct(void* y,void* w,void* q,int R,int K,void* stream) {
  quadratic_direct<<<R,128,K*sizeof(float),(cudaStream_t)stream>>>((float*)y,(float*)w,(float*)q,K);
  return int(cudaPeekAtLastError());
}
