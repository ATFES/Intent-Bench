#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    omp_set_dynamic(0);
    omp_set_num_threads(atoi(argv[1]));
    const size_t n = 1<<24;
    float *x = aligned_alloc(256,n*sizeof(float));
    if (!x) return 1;
    volatile double sink=0;
    #pragma omp parallel for
    for(size_t i=0;i<n;i++) x[i]=(float)(i%17);
    for(int trial=0;trial<9;trial++){
        for(int stride=1;stride<=16;stride*=16){
            double sum=0, start=now();
            #pragma omp parallel for simd reduction(+:sum) schedule(static)
            for(size_t i=0;i<n/stride;i++) sum+=x[i*stride];
            double stop=now();sink+=sum;
            printf("read_%d %.9g\n",stride,n/stride*sizeof(float)/(stop-start));
            start=now();
            #pragma omp parallel for simd schedule(static)
            for(size_t i=0;i<n/stride;i++) x[i*stride]=(float)(trial+1);
            stop=now();printf("write_%d %.9g\n",stride,n/stride*sizeof(float)/(stop-start));
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
