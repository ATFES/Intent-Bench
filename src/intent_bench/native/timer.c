#include <stdint.h>
#include <time.h>
typedef void (*operation)(int64_t, void*,void*,void*,void*,void*,void*,void*,void*,void*,int64_t,int64_t,int64_t);
double bench_time(operation fn, int64_t op, void **p, int64_t *s, int64_t repeat) {
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (int64_t i=0; i<repeat; ++i)
        fn(op,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],s[0],s[1],s[2]);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return ((end.tv_sec-begin.tv_sec)*1000.0+(end.tv_nsec-begin.tv_nsec)/1e6)/repeat;
}
