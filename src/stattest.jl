#####################################################
#%% Generate surrogates to test statistical significance
#####################################################

function fourier_surrogate(x::Vector{T}) where {T<:Real}
    return irfft( rfft(x) .* exp( 2 * π * rand() * im ), length(x) )
end

function fourier_surrogates(x::Vector{T}, ns::Int) where {T<:Real}
    S = zeros(ns, length(x))
    for i in axes(S, 1)
        S[i, :] = fourier_surrogate(x)
    end
    return S
end

function shuffle_surrogate()
    xshuffle = shuffle(x)
end

#####################################################
#%% Helpers to compute statistics
#####################################################

# function compute_stat(x::Vector{T}, S::Matrix{T}, slide_stat) where {T<:Real}
#     ref_stat = slide_stat(x)
#     sur_stat = vcat( mapslices( surrogate -> slide_stat(surrogate), S, dims=2) )
#     return ref_stat, sur_stat
# end

function slide_estimator(x::Vector{T}, hw::Int, estimator) where {T<:Real}
    nx = length(x)
    stat = fill(NaN, nx)
    for i in (hw+1):(nx-hw)
        x_windowed = centered_window( x, i, hw )
        stat[i] = estimator(x_windowed)
    end
    return stat
end

#####################################################
#%% Increase detection via Kendall-tau or regression
#####################################################

# Ridge regression with regularization parameter λ.
# Set to 0 to recover linear regression.
function ridge_regression(t::Vector{T}, x::Vector{T}, λ::Real) where {T<:Real}
    T_bias_ext = hcat( t, ones(length(t)) )'
    w = inv(T_bias_ext * T_bias_ext' + λ .* I(2)) * T_bias_ext * x
    return w
end

function slide_regression(
    t::Vector{T},
    x::Vector{T},
    hw::Int,
    λ::Real,
    pad_step::Int,
) where {T<:Real}

    w = fill(NaN, length(x))
    for i in (hw+1):pad_step:(nx-hw)
        x_windowed = centered_window( x, i, hw )
        t_windowed = centered_window( t, i, hw )
        w[i] = ridge_regression(t_windowed, x_windowed, λ)[1]
    end
    return w
end

function slide_regression(
    t::Vector{AbstractFloat},
    S::Matrix{T},
    hw::Int,
    λ::Real,
    pad_step::Int,
) where {T<:Real}
    return mapslices( x -> slide_regression(t, x, hw, λ, pad_step), S, dims = 2 )
end

function slide_kendall_tau(t::Vector{T}, x::Vector{T}, hw::Int, pad_step::Int) where {T<:Real}
    nx = length(x)
    kt = fill(NaN, nx)
    for i in (hw+1):pad_step:(nx-hw)
        x_windowed = centered_window( x, i, hw )
        t_windowed = centered_window( t, i, hw )
        kt[i] = corkendall( t_windowed, x_windowed )
    end
    return kt
end

function slide_kendall_tau(t::Vector{T}, S::Matrix{T}, hw::Int, pad_step::Int) where {T<:Real}
    return mapslices( x -> slide_kendall_tau(t, x, hw, pad_step), S, dims = 2 )
end

function percentile_significance(ref_stat::Vector{T}, sur_stat::Matrix{T}) where {T<:AbstractFloat}
    p = fill(NaN, length(ref_stat))
    for i in eachindex(ref_stat)
        if !isnan(ref_stat[i])
            p[i] = get_percentile( ref_stat[i], sur_stat[:,i] )
        end
    end
    return p
end

function get_percentile(ref_stat::T, sur_stat::Vector{T}) where {T<:Real}
    return sum(sur_stat .< ref_stat) / length(sur_stat)
end