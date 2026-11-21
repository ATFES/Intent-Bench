#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

static void load_only(float *p, size_t n, size_t step) {
#if defined(__AVX512F__)
    size_t i=0;
    if(step==1) {
        for(; i+16<=n; i+=16) {
            __m512 x=_mm512_loadu_ps(p+i);
            __asm__ volatile("" :: "v"(x));
        }
    }
    for(;i<n;i+=step) __asm__ volatile("" :: "x"(p[i]));
#elif defined(__riscv_vector)
    while(n) {
        size_t vl;
        __asm__ volatile("vsetvli %0,%1,e32,m4,ta,ma" : "=r"(vl) : "r"(n/step));
        if(step==1) __asm__ volatile("vle32.v v8,(%0)" :: "r"(p) : "memory");
        else __asm__ volatile("vlse32.v v8,(%0),%1" :: "r"(p),"r"(4*step) : "memory");
        p+=vl*step; n-=vl*step;
    }
#else
    for(size_t i=0;i<n;i+=step) (void)((volatile float *)p)[i];
#endif
}

static void bandwidth(float *x, size_t bytes, const char *tier, int trial) {
    size_t n=bytes/sizeof(float);
    int trips=bytes<1048576 ? 64 : bytes<16*1048576 ? 8 : 1;
    for(int stride=1;stride<=16;stride*=16) for(int side=0;side<2;side++) {
        int write=side==(trial&1);
        double begin=omp_get_wtime();
        #pragma omp parallel
        {
            size_t t=omp_get_thread_num(), nt=omp_get_num_threads();
            size_t first=(n/stride*t/nt)*stride, stop=(n/stride*(t+1)/nt)*stride;
            for(int r=0;r<trips;r++){
                if(!write)load_only(x+first,stop-first,stride);
                else {
                    #pragma omp simd
                    for(size_t i=first;i<stop;i+=stride) x[i]=1.0f;
                    __asm__ volatile("" ::: "memory");
                }
            }
        }
        printf("%s_%d_%s_bw %.9g\n",write?"write":"read",stride,tier,
               (double)bytes/stride*trips/((omp_get_wtime()-begin)*1e9));
    }
}

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    omp_set_dynamic(0);
    omp_set_num_threads(atoi(argv[1]));
    const size_t n = (sizeof(void*)==8 &&
#if defined(__x86_64__)
                        1
#else
                        0
#endif
                        ) ? (1ul<<28) : (1ul<<26);
    float *x = aligned_alloc(256,n*sizeof(float));
    if (!x) return 1;
    volatile double sink=0;
    #pragma omp parallel for
    for(size_t i=0;i<n;i++) x[i]=(float)(i%17);
    for(int trial=0;trial<9;trial++){
        bandwidth(x,16*1024*atoi(argv[1]),"private",trial);
        bandwidth(x,512*1024*atoi(argv[1]),"mid",trial);
        bandwidth(x,16ul<<20,"shared",trial);
        bandwidth(x,n*sizeof(float),"dram",trial);
        // A paired binary16 copy changes only the side carrying the stride.
        // Each strided and each contiguous stream touches the same array on
        // gather as on scatter. The checksum keeps the measured writes live.
        uint16_t *sparse=(uint16_t *)x;
        for (int stride=1; stride<=256; stride*=16) {
            size_t count=(1ul<<24)/stride;
            uint16_t *dense=aligned_alloc(256,count*sizeof(uint16_t));
            if(!dense) return 1;
            for (size_t i=0;i<count;i++) dense[i]=(uint16_t)i;
            for(int side=0;side<2;side++){
                int gather=side==(trial&1);
                double t=now();
                if(gather){
                    #pragma omp parallel for simd schedule(static)
                    for(size_t i=0;i<count;i++) dense[i]=sparse[i*stride];
                }else{
                    #pragma omp parallel for simd schedule(static)
                    for(size_t i=0;i<count;i++) sparse[i*stride]=dense[i];
                }
                double elapsed=now()-t;
                printf("copy_%s_%d_bw %.9g\n",gather?"gather":"scatter",stride,count*sizeof(uint16_t)/elapsed);
                sink+=gather?dense[trial]:sparse[(size_t)trial*stride];
            }
            free(dense);
        }
        double begin=now();
        #pragma omp parallel
        { __asm__ volatile("" ::: "memory"); }
        printf("launch %.9g\n",now()-begin);
        double sync=0;
        #pragma omp parallel reduction(max:sync)
        { double t=now();
          for(int k=0;k<1000;k++){
            #pragma omp barrier
          }
          sync=(now()-t)/1000.;
        }
        printf("sync %.9g\n",sync);
        const int blocks=256, lanes=256, repeats=128;
        begin=now();
        #pragma omp parallel for schedule(static)
        for(int b=0;b<blocks;b++){
            float *a=x+b*lanes;
            for(int k=0;k<repeats;k++){
                #pragma omp simd
                for(int j=0;j<lanes;j++) a[j]=a[j]*1.0001f+.0001f;
            }
        }
        printf("vector %.9g\n",2.*blocks*lanes*repeats/(now()-begin));
        sink+=x[trial];
    }
    free(x);return sink==0;
}
