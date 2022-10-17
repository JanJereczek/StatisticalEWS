using BenchmarkTools

function compute_TIsignificance(
    t::Vector{T},
    X::CuArray{T, 2},
    pindctr::WindowingParams,
    pidtrend::WindowingParams,
    ns::Int,
    ti_function_list = [ar1_whitenoise, cuvar],
    trend_measure_function = ridge_regression_slope,
    window = centered_window;
    kwargs...,
) where {T}

    nx, nt = size(X)
    S = generate_stacked_fourier_surrogates(X, ns)
    ti_labels = string.(ti_function_list)

    tindctr = trim_wndw( t, pindctr, window)
    tidtrend = trim_wndw( tindctr, pidtrend, window )

    significance = Dict{String, CuArray{T,2}}()

    for (ti_function, label) in zip(ti_function_list, ti_labels)

        reference_ti = slide_estimator( X, pindctr, ti_function, window )
        surrogate_ti = slide_estimator( S, pindctr, ti_function, window )

        reference_idtrend = slide_idtrend(reference_ti, tindctr, pidtrend, trend_measure_function, window; kwargs...)
        surrogate_idtrend = slide_idtrend(surrogate_ti, tindctr, pidtrend, trend_measure_function, window; kwargs...)

        significance[label] = get_percentile( reference_idtrend, surrogate_idtrend, ns, nx )
    end

    return significance, tidtrend
end

function benchmark_cpu_vs_gpu(Xcpu::Matrix{T}, Xgpu::CuArray{T, 2}, func) where {T}
    bmcpu = @benchmark $func( $Xcpu )
    bmgpu = @benchmark $func( $Xgpu )
    return [StatsBase.mean(bmcpu.times), StatsBase.mean(bmgpu.times)]
end

function benchmark_functions_fixedsize(Xcpu::Matrix{T}, function_list::Vector{Function}) where {T}
    nf = length(function_list)
    Xgpu = CuArray(Xcpu)
    bm_times = zeros(2, nf)
    for i in eachindex(function_list)
        bm_times[:, i] = benchmark_cpu_vs_gpu(Xcpu, Xgpu, function_list[i])
    end
    return bm_times
end

function benchmark_functions_over_size(T, nt::Int, nl_list::Vector{Int}, function_list::Vector{Function})
    nf = length(function_list)
    nd = length(nl_list)

    speedup_ratio = zeros(T, nf, nd)
    for j in eachindex(nl_list)
        Xcpu = rand(T, nl_list[j], nt)
        Xgpu = CuArray(Xcpu)
        for i in eachindex(function_list)
            bmtime = benchmark_cpu_vs_gpu(Xcpu, Xgpu, function_list[i])
            speedup_ratio[i, j] = bmtime[1] / bmtime[2]
        end
    end
    return speedup_ratio
end

function barplot_benchmark(bmtimes::Matrix{T}, function_list) where {T}
    labels = string.(function_list)
    idx = 1:size(bmtimes, 2)
    x = repeat(idx, inner=(2,))
    y = vec(bmtimes ./ repeat(bmtimes[1, :]', 2) )
    alternate = vec(repeat(1:2, inner=(1,idx[end])))

    fig = Figure()
    ax = Axis(fig[1,1], title = L"Ratio of CPU vs. GPU run-time $\,$", xticks = (idx, labels))
    barplot!(ax, x, y, color = alternate, dodge = alternate)
    return fig, ax
end

function lineplot_benchmark(nl_list::Vector{Int}, speedup_ratio, function_list::Vector{Function})
    labels = string.(function_list)
    fig = Figure()
    ax = Axis(fig[1,1], title = L"Ratio of CPU vs. GPU run-time $\,$", xscale = log10, yscale = log10)
    for i in axes(speedup_ratio, 1)
        scatterlines!(ax, nl_list, speedup_ratio[i, :], label = labels[i])
    end
    hlines!(ax, [1f0], label = "Factor 1", color = :red)
    axislegend(ax, position = :lt)
    return fig, ax
end


function benchmark_ti_cpu_vs_gpu(
    t::Vector{T},
    X::CuArray{T, 2},
    pindctr::WindowingParams,
    pidtrend::WindowingParams,
    ns::Int;
    ti_function_list = [ar1_whitenoise, cuvar],
    trend_measure_function = ridge_regression_slope,
    window = centered_window,
    kwargs...,
) where {T}

    nti = length(ti_function_list)
    labels = string.(ti_function_list)
    cpu_time = zeros(nti)
    gpu_time = zeros(nti)

    for 
        bm = @benchmark compute_TIsignificance(t, X, pindctr, pidtrend, ns, ti_function_list, trend_measure_function, window)
    end
end

# struct SolutionStruct
#     S
#     reference_ti
#     surrogate_ti