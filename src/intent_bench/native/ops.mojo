from std.math import exp, sqrt

comptime F = Pointer[Float32, MutUntrackedOrigin]
comptime I = Pointer[Int64, MutUntrackedOrigin]


@export("bench_run")
def bench_run(op: Int64, a: F, b: F, c: F, d: F, output: F, out2: F, out3: F,
              indices: I, offsets: I, rows: Int64, columns: Int64, count: Int64) abi("C"):
    var m = Int(rows)
    var n = Int(columns)
    var size = Int(count)
    if op == 0:  # ReLU
        for i in range(size):
            var x = a[unsafe_offset=i]
            output[unsafe_offset=i] = max(x, Float32(0))
    elif op == 1:  # SwiGLU
        for i in range(size):
            var x = a[unsafe_offset=i]
            output[unsafe_offset=i] = x / (1 + exp(-x)) * b[unsafe_offset=i]
    elif op == 2:  # transpose, independent tiled loops
        for rb in range(0, m, 32):
            for cb in range(0, n, 32):
                for r in range(rb, min(rb + 32, m)):
                    for col in range(cb, min(cb + 32, n)):
                        output[unsafe_offset=col*m+r] = a[unsafe_offset=r*n+col]
    elif op == 3:  # embedding
        for r in range(size):
            var idx = Int(indices[unsafe_offset=r])
            for col in range(n):
                output[unsafe_offset=r*n+col] = a[unsafe_offset=idx*n+col]
    elif op == 4:  # segmented mean
        for r in range(m):
            var begin = Int(offsets[unsafe_offset=r])
            var end = Int(offsets[unsafe_offset=r+1])
            for col in range(n):
                var total = Float32(0)
                for i in range(begin, end):
                    total += a[unsafe_offset=i*n+col]
                output[unsafe_offset=r*n+col] = total / Float32(max(1, end-begin))
    elif op == 5:  # ordered row prefix sum
        for r in range(m):
            var total = Float32(0)
            for col in range(n):
                total += a[unsafe_offset=r*n+col]
                output[unsafe_offset=r*n+col] = total
    elif op == 6:  # CSR SpMV
        for r in range(m):
            var total = Float32(0)
            for j in range(Int(offsets[unsafe_offset=r]), Int(offsets[unsafe_offset=r+1])):
                total += a[unsafe_offset=j] * b[unsafe_offset=Int(indices[unsafe_offset=j])]
            output[unsafe_offset=r] = total
    elif op == 7:  # histogram over explicit integer bins
        for bin in range(n):
            output[unsafe_offset=bin] = 0
        for i in range(size):
            var bin = Int(indices[unsafe_offset=i])
            if bin >= 0 and bin < n:
                output[unsafe_offset=bin] += 1
    elif op == 8:  # AdamW, beta1=.9, beta2=.999, eps=1e-8, lr=.001, wd=.01
        for i in range(size):
            var g = b[unsafe_offset=i]
            var first = Float32(0.9)*c[unsafe_offset=i] + Float32(0.1)*g
            var second = Float32(0.999)*d[unsafe_offset=i] + Float32(0.001)*g*g
            out2[unsafe_offset=i] = first
            out3[unsafe_offset=i] = second
            output[unsafe_offset=i] = a[unsafe_offset=i]*Float32(1-0.001*0.01) - Float32(0.001)*(first/Float32(0.1))/(sqrt(second/Float32(0.001))+Float32(1e-8))
