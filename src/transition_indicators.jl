using Statistics, StatsBase, FFTW, SparseArrays

#####################################################
#%% Statistical moments
#####################################################

# Mean on GPU.
function mean(X::CuArray{T, 2}) where {T<:Real}
    return reduce( +, X, dims=2) ./ T(size(X, 2))
end

# Accelerated masked mean on GPU.
function mean(
    X::CuArray{T, 2},       # Array of which mean should be computed.
    M::CuArray{T, 2},       # Mask matrix.
    pwin::WindowingParams,  # Windowing parameter struct.
) where {T<:Real}
    return (X * M) ./ (2*pwin.N_indctr_wndw+1)
end

# Variance on CPU.
function var(x::Vector{T}) where {T<:Real}
    return StatsBase.var(x)
end

# Variance on CPU.
function var(x::Matrix{T}) where {T<:Real}
    return StatsBase.var(x, dims=2)
end

# Variance on GPU with provided mean.
function var(X::CuArray{T, 2}, x_mean::CuArray{T, 2}) where {T<:Real}
    return reduce( +, (X .- x_mean).^2, dims=2) ./ T(size(X, 2) - 1)
end

# Variance on GPU.
function var(X::CuArray{T, 2}) where {T<:Real}
    x_mean = mean(X)
    return var(X, x_mean)
end

# Skewness on CPU.
function skw(x::Vector{T}) where {T<:Real}
    return StatsBase.skewness(x)
end

function skw(X::Matrix{T}, x_mean::Matrix{T}, x_var::Matrix{T}) where {T<:Real}
    return reduce( +, (X .- x_mean).^3, dims=2) ./ size(X, 2) ./ (x_var .^ T(1.5))
end

function skw(X::Matrix{T}) where {T<:Real}
    x_mean = StatsBase.mean(X, dims=2)
    x_var = var(X)
    return skw(X, x_mean, x_var)
end

# Skewness on GPU with provided mean and variance.
function skw(X::CuArray{T, 2}, x_mean::CuArray{T, 2}, x_var::CuArray{T, 2}) where {T<:Real}
    return reduce( +, (X .- x_mean).^3, dims=2) ./ size(X, 2) ./ (x_var .^ T(1.5))
end

# Skewness on GPU.
function skw(X::CuArray{T, 2}) where {T<:Real}
    x_mean = mean(X)
    x_var = var(X)
    return skw(X, x_mean, x_var)
end

# Kurtosis on CPU.
function krt(x::Vector{T}) where {T<:Real}
    return StatsBase.kurtosis(x)
end

function krt(X::Matrix{T}, x_mean::Matrix{T}, x_var::Matrix{T}) where {T<:Real}
    return reduce( +, (X .- x_mean).^3, dims=2) ./ size(X, 2) ./ (x_var .^ T(1.5))
end

function krt(X::Matrix{T}) where {T<:Real}
    x_mean = StatsBase.mean(X, dims=2)
    x_var = var(X)
    return reduce( +, (X .- x_mean).^3, dims=2) ./ size(X, 2) ./ (x_var .^ T(1.5))
end

# Kurtosis on GPU with provided mean and variance.
function krt(X::CuArray{T, 2}, x_mean::CuArray{T, 2}, x_var::CuArray{T, 2}) where {T<:Real}
    return reduce( +, (X .- x_mean) .^ 4, dims=2 ) ./ size(X, 2) ./ (x_var .^ 2)
end

# Kurtosis on GPU.
function krt(X::CuArray{T, 2}) where {T<:Real}
    x_mean = mean(X)
    x_var = var(X)
    return krt(X, x_mean, x_var)
end

#####################################################
#%% AR1 model
#####################################################

# TODO right thorough tests for each function.

# AR1 coefficient of a vector x for white noise assumption.
# M. Mudelsee, Climate Time Series Analysis, eq 2.4
function ar1_whitenoise(x::Vector{T}) where {T<:Real}
    N = length(x)
    return (x[2:N]' * x[1:N-1]) / (x[2:N]' * x[2:N])
end

# AR1 coefficients for a vertical stack of time series X for white noise assumption.
function ar1_whitenoise(X::Array{T}) where {T<:Real}
    # mapslices( x_ -> ar1_whitenoise(x_), X, dims=2)
    return reduce( +, X[:, 2:end] .* X[:, 1:end-1], dims=2) ./ reduce( +, X[:, 2:end] .* X[:, 2:end], dims=2)
end

# GPU accelerated AR1 coefficients for a vertical stack of time series X for white noise assumption.
function ar1_whitenoise(X::CuArray{T, 2}) where {T<:Real}
    return reduce( +, X[:, 2:end] .* X[:, 1:end-1], dims=2) ./ reduce( +, X[:, 2:end] .* X[:, 2:end], dims=2)
end

# Compute mask (representing window sliding) used for some acclerated computation on GPU.
function gpuMask(X::CuArray{T, 2}, pwin::WindowingParams) where {T<:Real}
    nt = size(X, 2)
    M = CuArray( diagm([i => ones(T, nt-1-abs(i)) for i in -pwin.N_indctr_wndw:pwin.N_indctr_wndw]...) )
    return M
end

# Masked version of GPU-accelerated computation of AR1 with sliding window.
function ar1_whitenoise(X::CuArray{T, 2}, M::CuArray{T, 2}) where {T<:Real}
    return ( (X[:, 2:end] .* X[:, 1:end-1]) * M ) ./ ( (X[:, 2:end] .^ 2) * M )
end

# Masked version of CPU-sparse computation of AR1 with sliding window.
function ar1_whitenoise(X::Matrix{T}, pwin) where {T<:Real}
    nt = size(X, 2)
    M = spdiagm([i => ones(T, nt-1-abs(i)) for i in -pwin.N_indctr_wndw:pwin.N_indctr_wndw]...)
    return ( (X[:, 2:end] .* X[:, 1:end-1]) * M ) ./ ( (X[:, 2:end] .^ 2) * M )
end


# TODO implement the TIs below.

# Restoring rate on CPU.
# assumes that noise has constant AC, not variance!
function restoring_rate()
end

# Restoring rate estimated by generalised least square on CPU.
# Works even for noise with varying AC and variance!
function restoring_rate_gls()
end

#####################################################
#%% Frequency spectrum
#####################################################

# Compute low-frequency power spectrum on CPU.
function lfps(x::Vector{T}; q_lowfreq=0.1) where {T<:Real}
    Psymmetrical = abs.(rfft(x) .^ 2)
    N = length(Psymmetrical)
    P = Psymmetrical[ roundint(N/2+1):end ]
    n = length(P)
    Pnorm = P ./ sum(P)
    nlow = roundint( n * q_lowfreq )
    return sum( Pnorm[1:nlow] )
end

function lfps(X::Matrix{T}; q_lowfreq=0.1) where {T<:Real}
    P = abs.(rfft(X, 2)) # .^ 2
    Pnorm = P ./ reduce(+, P, dims=2)
    return reduce(+, Pnorm[:, 1:roundint(q_lowfreq * size(Pnorm, 2))], dims=2)
end

function lfps(X::CuArray{T, 2}; q_lowfreq=0.1) where {T<:Real}
    P = abs.(CUDA.CUFFT.rfft( X, 2 )) # .^2
    Pnorm = P ./ reduce(+, P, dims=2)
    return reduce(+, Pnorm[:, 1:roundint(q_lowfreq * size(Pnorm, 2))], dims=2)
end

#####################################################
#%% Spatial indicators
#####################################################

# TODO implement EWSs below
function network_connectivity()
end

function spatial_variance()
end

function spatial_ar1()
end

function recovery_length()
end

function density_ratio()
end

#####################################################
#%% Sliding indicators
#####################################################

# Mutliple dispatch:
# - if windowing not specified, centered window is taken.

function slide_estimator(
    x::Vector{T},
    p::WindowingParams,
    estimator,
    wndw,
) where {T<:Real}

    nt = length(x)
    strided_idx = wndw(p.Nwndw, p.Nstrd, nt)
    nidx = length(strided_idx)

    transition_indicator = fill(T(NaN), nidx)
    for j1 in eachindex(strided_idx)
        j2 = strided_idx[j1]
        transition_indicator[j1] = estimator( wndw( x, j2, p.Nwndw ) )
    end
    return transition_indicator
end

function slide_estimator(
    X::Matrix{T},
    p::WindowingParams,
    estimator,
    wndw,
) where {T<:Real}

    nl, nt = size(X)
    strided_idx = wndw(p.Nwndw, p.Nstrd, nt)
    nidx = length(strided_idx)

    transition_indicator = fill(T(NaN), nl, nidx)
    for j1 in eachindex(strided_idx)
        j2 = strided_idx[j1]
        transition_indicator[:, j1] = estimator( wndw( X, j2, p.Nwndw ) )
    end
    return transition_indicator
end

function slide_estimator(
    X::CuArray{T, 2},
    p::WindowingParams,
    estimator,
    wndw,
) where {T<:Real}

    nl, nt = size(X)
    strided_idx = wndw(p.Nwndw, p.Nstrd, nt)
    nidx = length(strided_idx)

    transition_indicator = fill(T(NaN), nl, nidx)
    for j1 in eachindex(strided_idx)
        j2 = strided_idx[j1]
        transition_indicator[:, j1] = Array( estimator( wndw( X, j2, p.Nwndw ) ) )
    end
    return CuArray( transition_indicator )
end