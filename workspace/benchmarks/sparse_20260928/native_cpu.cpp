#include <cmath>
#include <omp.h>
extern "C" void dbnn_cpu_scan(const float* x,float* y,const float* da,
 const float* dc,const float* om,int B,int K,int T,int threads) {
  #pragma omp parallel for num_threads(threads)
  for(int bn=0;bn<B*K;bn++) {
    int b=bn/K,n=bn%K;
    float sa=0,sc=0;
    for(int t=0;t<T;t++) {
      float v=x[bn*T+t]; sa=da[n]*sa+v;sc=dc[n]*sc+v;
      y[(b*T+t)*K+n]=om[n]*(sa-sc);
    }
  }
}
extern "C" void dbnn_cpu_reset(const float* d,float* v,int B,int T,
 float alpha,float amp,float vth,int threads) {
  #pragma omp parallel for num_threads(threads)
  for(int b=0;b<B;b++) {
    float r=0;
    for(int t=0;t<T;t++) {
      float out=d[b*T+t]+r;v[b*T+t]=out;
      r=alpha*r-amp*(out>=vth);
    }
  }
}
