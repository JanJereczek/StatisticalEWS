function compute_TIsignificance(
    t::Vector{T},
    X::CuArray{T, 2},
    pindctr::WindowingParams,
    pidtrend::WindowingParams,
    ti_function_list,
    trend_measure_function,
    window,
    ns::Int;
    kwargs...,
)

    nx, nt = size(X)
    S = generate_stacked_fourier_surrogates(X, ns)
    ti_labels = string.(ti_function_list)

    tindctr = trim_wndw( t, pindctr, window)
    tidtrend = trim_wndw( tindctr, pidtrend, window )

    significance = Dict{String, CuArray{T,2}}()

    for (ti_function, label) in zip(ti_function_list, ti_labels)

        reference_ti = slide_estimator( X, pindctr, ti_function, window )
        surrogate_ti = slide_estimator( S, pindctr, ti_function, window )

        reference_idtrend = slide_idtrend(reference_ti, tindctr, pidtrend, ridge_regression_slope, window; kwargs...)
        surrogate_idtrend = slide_idtrend(surrogate_ti, tindctr, pidtrend, ridge_regression_slope, window; kwargs...)

        significance[label] = get_percentile( reference_idtrend, surrogate_idtrend, ns, nx )
    end

    return significance, tidtrend
end

# struct SolutionStruct
#     S
#     reference_ti
#     surrogate_ti