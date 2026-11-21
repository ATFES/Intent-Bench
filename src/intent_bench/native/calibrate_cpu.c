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

static void transfer(float *p, size_t n, size_t step, int write) {
#if defined(__AVX512F__)
    size_t i=0;
    const __m512 ones=_mm512_set1_ps(1.f);
    if(step==1) {
        for(; i+16<=n; i+=16) {
            if (write) _mm512_storeu_ps(p+i, ones);
            else {
                __m512 x=_mm512_loadu_ps(p+i);
                __asm__ volatile("" :: "v"(x));
            }
        }
    } else {
        const __m512i offsets=_mm512_mullo_epi32(_mm512_setr_epi32(
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), _mm512_set1_epi32(step));
        for(; n-i>=16*step; i+=16*step) {
            if (write) _mm512_i32scatter_ps(p+i,offsets,ones,4);
            else {
                __m512 x=_mm512_i32gather_ps(offsets,p+i,4);
                __asm__ volatile("" :: "v"(x));
            }
        }
    }
    for(;i<n;i+=step) {
        if(write) p[i]=1.f;
        else __asm__ volatile("" :: "x"(p[i]));
    }
#elif defined(__riscv_vector)
    size_t remaining=n/step;
    while(remaining) {
        size_t vl;
        __asm__ volatile("vsetvli %0,%1,e32,m4,ta,ma" : "=r"(vl) : "r"(remaining));
        if(write){
            if(step==1) __asm__ volatile("vfmv.v.f v8,%1\n\tvse32.v v8,(%0)" :: "r"(p),"f"(1.f) : "memory");
            else __asm__ volatile("vfmv.v.f v8,%1\n\tvsse32.v v8,(%0),%2" :: "r"(p),"f"(1.f),"r"(4*step) : "memory");
        }
        else if(step==1) __asm__ volatile("vle32.v v8,(%0)" :: "r"(p) : "memory");
        else __asm__ volatile("vlse32.v v8,(%0),%1" :: "r"(p),"r"(4*step) : "memory");
        remaining-=vl; if(remaining) p+=vl*step;
    }
#else
    for(size_t i=0;i<n;i+=step) {
        if(write)((volatile float *)p)[i]=1.f;
        else (void)((volatile float *)p)[i];
    }
#endif
    __asm__ volatile("" ::: "memory");
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
                transfer(x+first,stop-first,stride,write);
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
        // Visit short columns of a synthetic bank, then advance the starting
        // lane. The gather and scatter experiments use the same addresses and
        // demand copies. Adjacent workers can touch the same cache line.
        for(int step=32;step<=512;step*=4) {
            uint16_t *bank=(uint16_t*)x;
            size_t block=(size_t)step*64, chunks=(1u<<20)/block;
            uint16_t *buffer=aligned_alloc(256,2*block*chunks);
            if(!buffer)return 1;
            for(size_t i=0;i<block*chunks;i++)buffer[i]=(uint16_t)i;
            for(int side=0;side<2;side++) {
                int gather=side==(trial&1);
                double start=omp_get_wtime();
                #pragma omp parallel for schedule(static)
                for(int lane=0;lane<step;lane++) {
                    for(size_t b=0;b<chunks;b++) {
                        uint16_t *s=bank+b*block+lane;
                        uint16_t *d=buffer+b*block+lane*64;
                        if(gather) {
                            #pragma omp simd
                            for(int j=0;j<64;j++)d[j]=s[j*step];
                        } else {
                            #pragma omp simd
                            for(int j=0;j<64;j++)s[j*step]=d[j];
                        }
                    }
                }
                double elapsed=omp_get_wtime()-start;
                printf("panel_%s_%d_bw %.9g\n",gather?"gather":"scatter",step,2.0*block*chunks/(elapsed*1e9));
                sink+=gather?buffer[trial]:bank[trial];
            }
            free(buffer);
        }
        // Measure a sequence of adjacent spans before advancing to the
        // next page bank. Odd pitches also sample non-power-of-two set maps.
        const int pitches[]={31,32,127,128,511,512};
        for(size_t pitch=0;pitch<sizeof(pitches)/sizeof(*pitches);pitch++) {
            int step=pitches[pitch];
            for(int span=32;span<=1024;span=(span==32?256:span*4)) {
                size_t bank_words=(size_t)step*span;
                int banks=(1u<<21)/bank_words;
                uint16_t *scattered=(uint16_t*)x;
                uint16_t *packed=malloc(2*bank_words*banks);
                if(!packed)return 1;
                for(size_t i=0;i<bank_words*banks;i++)packed[i]=(uint16_t)i;
                for(int side=0;side<2;side++){
                    int gather=side==(trial&1);
                    double t=omp_get_wtime();
                    #pragma omp parallel for schedule(static)
                    for(int task=0;task<banks*step;task++) {
                        size_t block=task/step, lane=task%step;
                        uint16_t *s=scattered+block*bank_words+lane;
                        uint16_t *d=packed+block*bank_words+lane*span;
                        if(gather) {
                            #pragma omp simd
                            for(int j=0;j<span;j++)d[j]=s[j*step];
                        }else{
                            #pragma omp simd
                            for(int j=0;j<span;j++)s[j*step]=d[j];
                        }
                    }
                    double elapsed=omp_get_wtime()-t;
                    printf("span_%s_%d_%d_bw %.9g\n",gather?"gather":"scatter",step,span,2.*bank_words*banks/(elapsed*1e9));
                    sink+=gather?packed[trial]:scattered[trial];
                }
                free(packed);
            }
        }
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
