#include <cuda_runtime.h>
#include <cub/cub.cuh>

// M time rows share each W tile. The GEMM accumulator is contracted with Y
// before leaving registers; only one scalar per row / column tile is stored.
template<int BM,int BN,int BK,int TM,int TN,int THREAD_N,bool TRI>
__global__ void tiled_drive(const float* __restrict__ Y,
                            const float* __restrict__ W,
                            float* __restrict__ partial,
                            int M,int A,int NT) {
  constexpr int THREAD_M=BM/TM, THREADS=THREAD_M*THREAD_N;
  static_assert(BN==THREAD_N*TN);
  __shared__ float ys[BM][BK];
  __shared__ float ws[BK][BN];
  int tid=threadIdx.x, tm=tid/THREAD_N, tn=tid%THREAD_N;
  int m0=blockIdx.x*BM, n0=blockIdx.y*BN;
  float acc[TM][TN]={};
  int kend=TRI ? min(A,n0+BN):A;
  for(int k0=0;k0<kend;k0+=BK) {
    for(int z=tid;z<BM*BK;z+=THREADS) {
      int m=z/BK,k=z%BK;
      ys[m][k]=(m0+m<M && k0+k<A)?Y[(m0+m)*A+k0+k]:0.f;
    }
    for(int z=tid;z<BK*BN;z+=THREADS) {
      int k=z/BN,n=z%BN;
      ws[k][n]=(k0+k<A && n0+n<A)?W[(k0+k)*A+n0+n]:0.f;
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<BK;k++) {
      float av[TM],bv[TN];
      #pragma unroll
      for(int i=0;i<TM;i++)av[i]=ys[tm*TM+i][k];
      #pragma unroll
      for(int j=0;j<TN;j++)bv[j]=ws[k][tn*TN+j];
      #pragma unroll
      for(int i=0;i<TM;i++) {
        #pragma unroll
        for(int j=0;j<TN;j++)acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<TM;i++) {
    int m=m0+tm*TM+i;
    float value=0;
    #pragma unroll
    for(int j=0;j<TN;j++) {
      int n=n0+tn*TN+j;
      float y=(m<M && n<A)?Y[m*A+n]:0.f;
      value=fmaf(acc[i][j],y,value)+y;
    }
    #pragma unroll
    for(int delta=THREAD_N/2;delta>0;delta/=2)
      value+=__shfl_down_sync(0xffffffff,value,delta,THREAD_N);
    if(tn==0 && m<M) partial[m*NT+blockIdx.y]=value;
  }
}

__global__ void finish_drive(const float* p,float* out,int M,int NT,float bias) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r>=M)return;
  float v=0;
  for(int n=0;n<NT;n++)v+=p[r*NT+n];
  out[r]=v+bias;
}

// Literal per-timepoint-thread candidate. Y is transposed for coalesced time
// accesses; its conversion is a separate preparation cost and also timed in the
// end-to-end wrapper. Only used in small/medium-A exploratory comparisons.
__global__ void thread_drive(const float* __restrict__ yt,
                             const float* __restrict__ upper,
                             float* out,int M,int A,float bias) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r>=M)return;
  float q0=0,q1=0,q2=0,q3=0,linear=0;
  for(int j=0;j<A;j++) {
    float yj=yt[j*M+r];linear+=yj;
    float z0=0,z1=0,z2=0,z3=0;
    int i=0;
    for(;i+3<=j;i+=4) {
      z0=fmaf(yt[(i+0)*M+r],upper[(i+0)*A+j],z0);
      z1=fmaf(yt[(i+1)*M+r],upper[(i+1)*A+j],z1);
      z2=fmaf(yt[(i+2)*M+r],upper[(i+2)*A+j],z2);
      z3=fmaf(yt[(i+3)*M+r],upper[(i+3)*A+j],z3);
    }
    for(;i<=j;i++)z0=fmaf(yt[i*M+r],upper[i*A+j],z0);
    q0=fmaf(z0,yj,q0);q1=fmaf(z1,yj,q1);
    q2=fmaf(z2,yj,q2);q3=fmaf(z3,yj,q3);
  }
  out[r]=((q0+q1)+(q2+q3))+linear+bias;
}

__global__ void original_direct_fma(const float* y,const float* w,float* d,int M,int A,float bias) {
  using Reduce=cub::BlockReduce<float,128>;
  __shared__ typename Reduce::TempStorage temp;
  extern __shared__ float ys[];
  int r=blockIdx.x;
  for(int k=threadIdx.x;k<A;k+=128)ys[k]=y[r*A+k];
  __syncthreads();
  float total=0;
  for(int i=threadIdx.x;i<A;i+=128) {
    float sum=0;
    for(int j=0;j<A;j++)sum=fmaf(ys[j],w[j*A+i],sum);
    total=fmaf(ys[i],sum,total)+ys[i];
  }
  total=Reduce(temp).Sum(total);
  if(threadIdx.x==0)d[r]=total+bias;
}

template<int ROWS>
__global__ void warp_drive(const float* __restrict__ y,const float* __restrict__ w,float* out,int M,int A,float bias) {
  extern __shared__ float ys[];
  int warp=threadIdx.x/32,lane=threadIdx.x%32,r=blockIdx.x*ROWS+warp;
  for(int k=lane;k<A;k+=32)ys[warp*A+k]=r<M?y[r*A+k]:0.f;
  __syncwarp();
  float total=0;
  for(int i=lane;i<A;i+=32) {
    float q=0;
    for(int j=0;j<A;j++)q=fmaf(ys[warp*A+j],w[j*A+i],q);
    total=fmaf(ys[warp*A+i],q,total)+ys[warp*A+i];
  }
  #pragma unroll
  for(int delta=16;delta>0;delta/=2)total+=__shfl_down_sync(0xffffffff,total,delta);
  if(lane==0&&r<M)out[r]=total+bias;
}

template<bool TRI>
int dispatch(const float* y,const float* w,float* p,float* out,int M,int A,float bias,int config,cudaStream_t stream) {
  int BN=64;
  #define RUN(ID,MM,NN,KK,TM,TN,NT) case ID: { BN=NN; \
    tiled_drive<MM,NN,KK,TM,TN,NT,TRI><<<dim3((M+MM-1)/MM,(A+NN-1)/NN),(MM/TM)*NT,0,stream>>>(y,w,p,M,A,(A+NN-1)/NN);break; }
  switch(config) {
    RUN(0,32,64,16,4,4,16)
    RUN(1,32,64,32,4,4,16)
    RUN(2,64,64,16,8,4,16)
    RUN(3,64,64,32,8,4,16)
    RUN(4,32,128,16,4,8,16)
    RUN(5,32,128,32,4,8,16)
    RUN(6,64,64,16,4,4,16)
    RUN(7,64,128,16,8,8,16)
    default:return 999;
  }
  #undef RUN
  finish_drive<<<(M+255)/256,256,0,stream>>>(p,out,M,(A+BN-1)/BN,bias);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_tiled(void* y,void* w,void* p,void* out,int M,int A,float bias,int config,int triangle,void* stream) {
  if(triangle)return dispatch<true>((float*)y,(float*)w,(float*)p,(float*)out,M,A,bias,config,(cudaStream_t)stream);
  return dispatch<false>((float*)y,(float*)w,(float*)p,(float*)out,M,A,bias,config,(cudaStream_t)stream);
}
extern "C" int dbnn_thread(void* yt,void* w,void* out,int M,int A,float bias,void* stream) {
  thread_drive<<<(M+127)/128,128,0,(cudaStream_t)stream>>>((float*)yt,(float*)w,(float*)out,M,A,bias);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_direct_fma(void* y,void* w,void* out,int M,int A,float bias,void* stream) {
  original_direct_fma<<<M,128,A*sizeof(float),(cudaStream_t)stream>>>((float*)y,(float*)w,(float*)out,M,A,bias);
  return int(cudaPeekAtLastError());
}
extern "C" int dbnn_warp(void* y,void* w,void* out,int M,int A,float bias,int config,void* stream) {
  if(config==0)warp_drive<4><<<(M+3)/4,128,4*A*sizeof(float),(cudaStream_t)stream>>>((float*)y,(float*)w,(float*)out,M,A,bias);
  else if(config==1)warp_drive<8><<<(M+7)/8,256,8*A*sizeof(float),(cudaStream_t)stream>>>((float*)y,(float*)w,(float*)out,M,A,bias);
  else warp_drive<16><<<(M+15)/16,512,16*A*sizeof(float),(cudaStream_t)stream>>>((float*)y,(float*)w,(float*)out,M,A,bias);
  return int(cudaPeekAtLastError());
}
