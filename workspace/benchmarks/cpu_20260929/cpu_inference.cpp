#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

// Prepared, single-window CPU inference. The caller owns dense N x T input and
// output. Active-site IDs and all checkpoint-dependent data are cached here.
using Gemm = void (*)(const char*, const char*, const int*, const int*,
                     const int*, const float*, const float*, const int*,
                     const float*, const int*, const float*, float*, const int*);
using MklThreads = int (*)(int);
struct Plan {
  int K, T, S;
  float bias, amp, alpha, threshold;
  std::vector<int64_t> ids;
  std::vector<float> da, dc, omega, w, upper, pa, pc, reset_powers;
  std::vector<float> y, z, drive;
  Gemm gemm;
  MklThreads mkl_threads;
};

static inline __mmask16 mask16(int n) {
  return n >= 16 ? 0xffff : (__mmask16)((1u << n)-1);
}
static inline __m512 prefix16(__m512 x, float b) {
  const __m512i i1=_mm512_setr_epi32(0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
  const __m512i i2=_mm512_setr_epi32(0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13);
  const __m512i i4=_mm512_setr_epi32(0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11);
  const __m512i i8=_mm512_setr_epi32(0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7);
  x=_mm512_fmadd_ps(_mm512_maskz_permutexvar_ps(0xfffe,i1,x),_mm512_set1_ps(b),x);
  b*=b;
  x=_mm512_fmadd_ps(_mm512_maskz_permutexvar_ps(0xfffc,i2,x),_mm512_set1_ps(b),x);
  b*=b;
  x=_mm512_fmadd_ps(_mm512_maskz_permutexvar_ps(0xfff0,i4,x),_mm512_set1_ps(b),x);
  b*=b;
  return _mm512_fmadd_ps(_mm512_maskz_permutexvar_ps(0xff00,i8,x),_mm512_set1_ps(b),x);
}
static inline float last_lane(__m512 x) {
  return _mm_cvtss_f32(_mm512_castps512_ps128(_mm512_permutexvar_ps(_mm512_set1_epi32(15),x)));
}

static void filter(Plan& p,const float* x,int nt,int mode) {
  #pragma omp parallel for schedule(static) num_threads(nt)
  for(int k=0;k<p.K;++k) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    const float *in=x+p.ids[k]*p.T;
    float *out=p.y.data()+k*p.S, a=0,c=0;
    if(mode==0) {
      for(int t=0;t<p.T;++t) {
        a=std::fma(p.da[k],a,in[t]); c=std::fma(p.dc[k],c,in[t]);
        out[t]=p.omega[k]*(a-c);
      }
      std::fill(out+p.T,out+p.S,0.f);
    } else {
      const __m512 pa=_mm512_loadu_ps(p.pa.data()+16*k),pc=_mm512_loadu_ps(p.pc.data()+16*k);
      const __m512 om=_mm512_set1_ps(p.omega[k]);
      for(int t=0;t<p.T;t+=16) {
        const __mmask16 m=mask16(std::min(16,p.T-t));
        const __m512 v=_mm512_maskz_loadu_ps(m,in+t);
        __m512 va=_mm512_fmadd_ps(_mm512_set1_ps(a),pa,prefix16(v,p.da[k]));
        __m512 vc=_mm512_fmadd_ps(_mm512_set1_ps(c),pc,prefix16(v,p.dc[k]));
        _mm512_storeu_ps(out+t,_mm512_maskz_mov_ps(m,_mm512_mul_ps(om,_mm512_sub_ps(va,vc))));
        a=last_lane(va);c=last_lane(vc);
      }
    }
  }
}

template<int V> static void triangular_drive(Plan& p,int nt) {
  // SIMD lanes are physical time samples; OpenMP workers own disjoint time
  // tiles. Symmetrization counts every off-diagonal pair only once.
  #pragma omp parallel for schedule(static) num_threads(nt)
  for(int t=0;t<p.S;t+=16*V) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    __m512 sum[V];for(int v=0;v<V;++v)sum[v]=_mm512_set1_ps(p.bias);
    for(int i=0;i<p.K;++i) {
      __m512 inner[V];for(int v=0;v<V;++v)inner[v]=_mm512_setzero_ps();
      const float* wr=p.upper.data()+i*p.K;
      for(int j=i+1;j<p.K;++j) {
        const __m512 w=_mm512_set1_ps(wr[j]);
        for(int v=0;v<V;++v) {
          if(t+16*v<p.S)inner[v]=_mm512_fmadd_ps(w,_mm512_loadu_ps(p.y.data()+j*p.S+t+16*v),inner[v]);
        }
      }
      for(int v=0;v<V;++v) if(t+16*v<p.S) {
        const __m512 yi=_mm512_loadu_ps(p.y.data()+i*p.S+t+16*v);
        const __m512 diag=_mm512_mul_ps(_mm512_set1_ps(wr[i]),yi);
        sum[v]=_mm512_fmadd_ps(yi,_mm512_add_ps(_mm512_add_ps(inner[v],diag),_mm512_set1_ps(1.f)),sum[v]);
      }
    }
    for(int v=0;v<V;++v)if(t+16*v<p.S)_mm512_storeu_ps(p.drive.data()+t+16*v,sum[v]);
  }
}

static void blas_drive(Plan& p,int nt) {
  // Column-major view: Y is S x K, W is K x K. The resulting quadratic
  // contraction is invariant to transposing W. No low-precision arithmetic.
  const char n='N';const float one=1,zero=0;
  int old=p.mkl_threads(nt);
  p.gemm(&n,&n,&p.S,&p.K,&p.K,&one,p.y.data(),&p.S,p.w.data(),&p.K,&zero,p.z.data(),&p.S);
  p.mkl_threads(old);
  #pragma omp parallel for schedule(static) num_threads(nt)
  for(int t=0;t<p.S;t+=16) {
    __m512 q=_mm512_setzero_ps(),lin=_mm512_setzero_ps();
    for(int k=0;k<p.K;++k) {
      __m512 y=_mm512_loadu_ps(p.y.data()+k*p.S+t);
      q=_mm512_fmadd_ps(y,_mm512_loadu_ps(p.z.data()+k*p.S+t),q);
      lin=_mm512_add_ps(lin,y);
    }
    _mm512_storeu_ps(p.drive.data()+t,_mm512_add_ps(_mm512_add_ps(q,lin),_mm512_set1_ps(p.bias)));
  }
}

static int reset(Plan& p,float* out,int mode) {
  int count=0;
  if(mode==0) {
    float r=0;
    for(int t=0;t<p.T;++t) {
      float v=p.drive[t]+r;out[t]=v;bool s=v>=p.threshold;count+=s;
      r=p.alpha*r-(s?p.amp:0.f);
    }
    return count;
  }
  int start=0,t=0;float anchor=0;
  const __m512 th=_mm512_set1_ps(p.threshold);
  while(t<p.T) {
    const __mmask16 m=mask16(std::min(16,p.T-t));
    const __m512 r=_mm512_mul_ps(_mm512_set1_ps(anchor),_mm512_maskz_loadu_ps(m,p.reset_powers.data()+t-start));
    const __m512 v=_mm512_add_ps(_mm512_maskz_loadu_ps(m,p.drive.data()+t),r);
    unsigned hits=_mm512_cmp_ps_mask(v,th,_CMP_GE_OQ)&m;
    if(!hits) {_mm512_mask_storeu_ps(out+t,m,v);t+=16;}
    else {
      int at=__builtin_ctz(hits);
      _mm512_mask_storeu_ps(out+t,mask16(at+1),v);
      anchor=p.alpha*(anchor*p.reset_powers[t+at-start])-p.amp;
      start=t=t+at+1;++count;
    }
  }
  return count;
}

extern "C" {
void* dbnn_cpu_create(int K,int T,const int64_t* ids,const float* da,const float* dc,
                     const float* omega,const float* w,float bias,float amp,float alpha,
                     float threshold,void* gemm,void* mkl_threads) {
  auto* p=new Plan; p->K=K;p->T=T;p->S=(T+15)/16*16;
  p->bias=bias;p->amp=amp;p->alpha=alpha;p->threshold=threshold;
  p->ids.assign(ids,ids+K);p->da.assign(da,da+K);p->dc.assign(dc,dc+K);
  p->omega.assign(omega,omega+K);p->w.assign(w,w+K*K);p->upper.resize(K*K);
  p->pa.resize(K*16);p->pc.resize(K*16);p->reset_powers.resize(p->S);
  p->y.resize(K*p->S);p->z.resize(K*p->S);p->drive.resize(p->S);
  p->gemm=reinterpret_cast<Gemm>(gemm);p->mkl_threads=reinterpret_cast<MklThreads>(mkl_threads);
  for(int k=0;k<K;++k) {
    for(int j=k;j<K;++j)p->upper[k*K+j]=j==k?w[k*K+j]:w[k*K+j]+w[j*K+k];
    for(int i=0;i<16;++i){p->pa[16*k+i]=std::pow(double(da[k]),i+1);p->pc[16*k+i]=std::pow(double(dc[k]),i+1);}
  }
  for(int t=0;t<p->S;++t)p->reset_powers[t]=std::pow(double(alpha),t);
  return p;
}
void dbnn_cpu_destroy(void* ptr) {delete static_cast<Plan*>(ptr);}
int dbnn_cpu_forward(void* ptr,const float* dense,float* out,int nt,int filter_mode,
                     int drive_mode,int reset_mode,double* profile) {
  auto& p=*static_cast<Plan*>(ptr);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  auto now=[](){return std::chrono::steady_clock::now();};
  auto t0=now();filter(p,dense,nt,filter_mode);auto t1=now();
  if(drive_mode==0)blas_drive(p,nt);
  else if(drive_mode==1)triangular_drive<1>(p,nt);
  else triangular_drive<4>(p,nt);
  auto t2=now();int n=reset(p,out,reset_mode);auto t3=now();
  if(profile) {
    profile[0]=std::chrono::duration<double,std::milli>(t1-t0).count();
    profile[1]=std::chrono::duration<double,std::milli>(t2-t1).count();
    profile[2]=std::chrono::duration<double,std::milli>(t3-t2).count();
  }
  return n;
}
void dbnn_cpu_copy_drive(void* ptr,float* out) {
  auto& p=*static_cast<Plan*>(ptr);std::copy(p.drive.begin(),p.drive.begin()+p.T,out);
}
}
