#####################################################
#%% Generate surrogates to test statistical significance
#####################################################

# Shift each frequency content by a random phase.
function fourier_surrogate(x::Vector{T}) where {T<:Real}
    F = rfft(x)
    return irfft( F .* exp.( 2*π*im .* rand(length(F)) ), length(x) )
end

# Generate ns surrogates of x.
function fourier_surrogates(x::Vector{T}, ns::Int) where {T<:Real}
    S = zeros(T, ns, length(x))
    for i in axes(S, 1)
        S[i, :] = fourier_surrogate(x)
    end
    return S
end

function fourier_surrogates3D(X::Matrix{T}, ns::Int) where {T<:Real}
    nx, nt = size(X)
    S = zeros(T, nx, nt, ns)
    for i in 1:nx
        S[i, :, :] = fourier_surrogates(X[i, :], ns)'
    end
    return S
end

function fourier_surrogates2D(X::Matrix{T}, ns::Int) where {T<:Real}
    nx, nt = size(X)
    S = zeros(T, nx*ns, nt)
    for i in 1:nx
        S[(i-1)*ns+1:i*ns, :] = fourier_surrogates(X[i, :], ns)
    end
    return S, nx, ns
end

struct surrogates2D{T<:Real}
    S::Matrix{T}
    nx::Int
    ns::Int
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

# GPU version of ridge regression. Y must be nt x ns.
function ridge_regression(t::Vector{T}, Y::CuArray{T, 2}, λ::Real) where {T<:Real}
    t = t .- t[1]
    T_bias_ext = hcat( t, ones(length(t)) )'
    U = CuArray( inv(T_bias_ext * T_bias_ext' + λ .* I(2)) )
    W = U * CuArray( T_bias_ext ) * Y
    return W
end

function ridge_regression_slope(t::Vector{T}, Y::CuArray{T, 2}, λ::Real) where {T<:Real}
    return permutedims( ridge_regression(t, Y, λ) )[:, 1]
end

function slide_regression(
    t::Vector{T},
    x::Vector{T},
    hw::Int,
    λ::Real,
    stride::Int,
) where {T<:Real}

    nx = length(x)
    w = fill(NaN, nx)
    for i in (hw+1):stride:(nx-hw)
        x_windowed = centered_window( x, i, hw )
        t_windowed = centered_window( t, i, hw )
        w[i] = ridge_regression(t_windowed, x_windowed, λ)[1]
    end
    return w
end

function slide_regression(
    t::Vector{T},
    S::Matrix{T},
    hw::Int,
    λ::Real,
    stride::Int,
) where {T<:Real}
    return mapslices( x -> slide_regression(t, x, hw, λ, stride), S, dims = 2 )
end

function slide_kendall_tau(t::Vector{T}, x::Vector{T}, hw::Int, stride::Int) where {T<:Real}
    nx = length(x)
    kt = fill(NaN, nx)
    for i in (hw+1):stride:(nx-hw)
        x_windowed = centered_window( x, i, hw )
        t_windowed = centered_window( t, i, hw )
        kt[i] = corkendall( t_windowed, x_windowed )
    end
    return kt
end

function slide_kendall_tau(t::Vector{T}, S::Matrix{T}, hw::Int, stride::Int) where {T<:Real}
    return mapslices( x -> slide_kendall_tau(t, x, hw, stride), S, dims = 2 )
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

function percentile_significance(ref_stat::CuArray{T, 2}, sur_stat::CuArray{T,2}) where {T<:Real}
    return reduce( +, sur_stat .< ref_stat, dims=1 ) ./ size(sur_stat, 1)
end

function get_percentile(ref_stat::T, sur_stat::Vector{T}) where {T<:Real}
    return sum(sur_stat .< ref_stat) / length(sur_stat)
end

function check_std_endpoint(x::Vector{T}, xwin::Vector{T}, std_tol::Real) where {T<:Real}
    return max(xwin) > std_tol * std(x)
end