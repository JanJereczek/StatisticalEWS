using CUDA, BenchmarkTools

function cpu_multiply_rands(n::Int)
    return rand(n,n) * rand(n,n)
end

function gpu_multiply_rands(n::Int)
    return CUDA.rand(n,n) * CUDA.rand(n,n)
end

ntest = 100000
@benchmark cpu_multiply_rands(ntest)