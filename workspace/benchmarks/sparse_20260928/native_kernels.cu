#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "cutlass/gemm/device/gemm.h"

// All entry points consume device pointers and the caller's CUDA stream.
// The synaptic recurrence is exact (up to FP32 rounding), without tail truncation.
struct PairAffine { float a,c,u,v; };
struct Compose {
  __device__ PairAffine operator()(PairAffine x, PairAffine y) const {
    return {y.a*x.a,y.c*x.c, y.a*x.u+y.u, y.c*x.v+y.v};
  }
};
template<int BLOCK>
__global__ void scan_kernel(const float* x, float* y, const float* da,
 const float* dc, const float* om, int K, int T) {
  using Scan = cub::BlockScan<PairAffine,BLOCK>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ float last_a, last_c;
  int n=blockIdx.x, b=blockIdx.y, tid=threadIdx.x;
  if(tid==0) {last_a=0;last_c=0;}
  __syncthreads();
  for(int base=0;base<T;base+=BLOCK) {
    int t=base+tid;
    float val=t<T?x[(b*K+n)*T+t]:0;
    PairAffine p={da[n],dc[n],val,val}, q;
    Scan(temp).InclusiveScan(p,q,Compose());
    float sa=q.u+q.a*last_a, sc=q.v+q.c*last_c;
    if(t<T) y[(b*T+t)*K+n]=om[n]*(sa-sc);
    __syncthreads();
    if(tid==min(BLOCK,T-base)-1) {last_a=sa;last_c=sc;}
    __syncthreads();
  }
}
__global__ void reduce_kernel(const float* y,const float* z,float* d,int R,int K,float bias) {
  using Reduce=cub::BlockReduce<float,128>;
  __shared__ typename Reduce::TempStorage temp;
  int r=blockIdx.x;
  float v=0;
  for(int k=threadIdx.x;k<K;k+=128) {float a=y[r*K+k];v+=a*z[r*K+k]+a;}
  v=Reduce(temp).Sum(v);
  if(threadIdx.x==0) d[r]=v+bias;
}
__global__ void direct_kernel(const float* y,const float* w,float* d,int R,int K,float bias) {
  using Reduce=cub::BlockReduce<float,128>;
  __shared__ typename Reduce::TempStorage temp;
  extern __shared__ float ys[];
  int r=blockIdx.x;
  for(int k=threadIdx.x;k<K;k+=128) ys[k]=y[r*K+k];
  __syncthreads();
  float total=0;
  for(int i=threadIdx.x;i<K;i+=128) {
    float sum=0;
    for(int j=0;j<K;j++) sum+=ys[j]*w[j*K+i];
    total+=ys[i]*sum+ys[i];
  }
  total=Reduce(temp).Sum(total);
  if(threadIdx.x==0)d[r]=total+bias;
}
__global__ void reset_kernel(const float* d,float* v,int B,int T,float alpha,float amp,float vth) {
  int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b>=B)return;
  float r=0;
  for(int t=0;t<T;t++) {
    float out=d[b*T+t]+r;
    v[b*T+t]=out;
    r=alpha*r-amp*(out>=vth);
  }
}
// Event-driven threshold search: exact closed-loop reset between successive spikes.
// No spike truncation. All timesteps are searched in parallel within one block.
__global__ void reset_event_kernel(const float* d,float* v,const float* ap,
  int T,float alpha,float amp,float vth) {
  using Reduce=cub::BlockReduce<int,256>;
  __shared__ typename Reduce::TempStorage tmp;
  __shared__ int start,first;
  __shared__ float r;
  int b=blockIdx.x,tid=threadIdx.x;
  if(tid==0){start=0;r=0;}
  __syncthreads();
  while(start<T) {
    int candidate=T;
    for(int t=start+tid;t<T;t+=256) {
      float out=d[b*T+t]+r*ap[t-start];
      if(out>=vth)candidate=min(candidate,t);
    }
    int earliest=Reduce(tmp).Reduce(candidate,cub::Min());
    if(tid==0)first=earliest;
    __syncthreads();
    for(int t=start+tid;t<T&&t<=first;t+=256)v[b*T+t]=d[b*T+t]+r*ap[t-start];
    __syncthreads();
    if(tid==0) {
      if(first<T)r=alpha*(r*ap[first-start])-amp;
      start=first+1;
    }
    __syncthreads();
  }
}
extern "C" int dbnn_scan(void* x,void* y,void* da,void* dc,void* om,int B,int K,int T,void* stream) {
  scan_kernel<128><<<dim3(K,B),128,0,(cudaStream_t)stream>>>((float*)x,(float*)y,(float*)da,(float*)dc,(float*)om,K,T);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_reduce(void* y,void* z,void* d,int R,int K,float bias,void* stream) {
  reduce_kernel<<<R,128,0,(cudaStream_t)stream>>>((float*)y,(float*)z,(float*)d,R,K,bias);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_direct(void* y,void* w,void* d,int R,int K,float bias,void* stream) {
  direct_kernel<<<R,128,K*sizeof(float),(cudaStream_t)stream>>>((float*)y,(float*)w,(float*)d,R,K,bias);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_reset(void* d,void* v,int B,int T,float alpha,float amp,float vth,void* stream) {
  reset_kernel<<<(B+31)/32,32,0,(cudaStream_t)stream>>>((float*)d,(float*)v,B,T,alpha,amp,vth);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_reset_event(void* d,void* v,void* ap,int B,int T,float alpha,float amp,float vth,void* stream) {
  reset_event_kernel<<<B,256,0,(cudaStream_t)stream>>>((float*)d,(float*)v,(float*)ap,T,alpha,amp,vth);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_cutlass(void* a,void* b,void* c,int M,int K,void* stream) {
  using Gemm=cutlass::gemm::device::Gemm<float,cutlass::layout::RowMajor,
     float,cutlass::layout::RowMajor,float,cutlass::layout::RowMajor,
     float,cutlass::arch::OpClassSimt,cutlass::arch::Sm80>;
  typename Gemm::Arguments args({M,K,K},{(float*)a,K},{(float*)b,K},
      {(float*)c,K},{(float*)c,K},{1.0f,0.0f});
  Gemm gemm;
  auto status=gemm(args,nullptr,(cudaStream_t)stream);
  return status==cutlass::Status::kSuccess?0:1000+int(status);
}
