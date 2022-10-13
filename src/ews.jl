using CUDA, BenchmarkTools

#####################################################
#%% Smoothing
#####################################################

# Rollmean with output length = input length.
function gettrend_rollmean(x::Vector{T}, hw::Int) where {T<:Real}
    xtrend = rollmean(x, 2*hw+1)
    # xtrend = vcat( fill(T(NaN), hw), xtrend, fill(T(NaN), hw) )
    # xtrend = vcat( x[1:hw], xtrend, x[end-hw+1:end] )
    return xtrend
end

function gettrend_gaussiankernel(x::Vector{T}, σ::Int) where {T<:Real}
    return imfilter( x, Kernel.gaussian((σ,)) )
end

# TODO implement further ways to detrend the signal
function gettrend_loess(x::Vector{T}, σ::Int) where {T<:Real}
    xtrend = x
    return xtrend
end

function gettrend_dfa(x::Vector{T}, σ::Int) where {T<:Real}
    xtrend = x
    return xtrend
end

function gettrend_emd(x::Vector{T}, σ::Int) where {T<:Real}
    xtrend = x
    return xtrend
end

# Remove trend.
function detrend(x::Vector{T}, x_trend::Vector{T}) where {T<:Real}
    return x - x_trend
end

#####################################################
#%% Statistical moments
#####################################################

# Mean on GPU (helper).
function cumean(X::CuArray{T, 2}) where {T<:Real}
    return reduce( +, X, dims=2) ./ T(size(X, 2))
end

# Mean on GPU (helper).
function cumean(X::CuArray{T, 2}, M::CuArray{T, 2}, pwin::WindowingParams) where {T<:Real}
    return (X * M) ./ (2*pwin.Nindctr+1)
end

# Variance on CPU.
function variance(x::Vector{T}) where {T<:Real}
    return StatsBase.var(x)
end

# Variance on GPU with provided mean.
function cuvar(X::CuArray{T, 2}, x_mean::CuArray{T, 2}) where {T<:Real}
    return reduce( +, (X .- x_mean).^2, dims=2) ./ T(size(X, 2) - 1)
end

# Variance on GPU.
function cuvar(X::CuArray{T, 2}) where {T<:Real}
    x_mean = cumean(X)
    return cuvar(X, x_mean)
end

# Skewness on CPU.
function skw(x::Vector{T}) where {T<:Real}
    return StatsBase.skewness(x)
end

# Skewness on GPU with provided mean and variance.
function cuskw(X::CuArray{T, 2}, x_mean::CuArray{T, 2}, x_var::CuArray{T, 2}) where {T<:Real}
    return reduce( +, (X .- x_mean).^3, dims=2) ./ size(X, 2) ./ (x_var .^ T(1.5))
end

# Skewness on GPU.
function cuskw(X::CuArray{T, 2}) where {T<:Real}
    x_mean = cumean(X)
    x_var = cuvar(X)
    return cuskw(X, x_mean, x_var)
end

# Kurtosis on CPU.
function krt(x::Vector{T}) where {T<:Real}
    return StatsBase.kurtosis(x)
end

# Kurtosis on GPU with provided mean and variance.
function cukrt(X::CuArray{T, 2}, x_mean::CuArray{T, 2}, x_var::CuArray{T, 2}) where {T<:Real}
    return reduce( +, (X .- x_mean) .^ 4, dims=2 ) ./ size(X, 2) ./ (x_var .^ 2)
end

# Kurtosis on GPU.
function cukrt(X::CuArray{T, 2}) where {T<:Real}
    x_mean = cumean(X)
    x_var = cuvar(X)
    return cukrt(X, x_mean, x_var)
end

#####################################################
#%% AR1 model
#####################################################

# Compute AR1 coefficient θ of a vector x for white noise assumption.
function ar1_whitenoise(x::Vector{T}) where {T<:Real}
    N = length(x)
    num = x[2:N]' * x[1:N-1]
    denum = x[2:N]' * x[2:N]
    θ = num / denum
    return θ
end

function ar1_whitenoise(X::Array{T}) where {T<:Real}
    return mapslices( x_ -> ar1_whitenoise(x_), X, dims=2)
end

function ar1_whitenoise(x::CuArray{T, 2}) where {T<:Real}
    return reduce( +, x[:, 2:end] .* x[:, 1:end-1], dims=2) ./ reduce( +, x[:, 2:end] .* x[:, 2:end], dims=2)
end

function gpuMask(
    X::CuArray{T, 2},
    pwin::WindowingParams,
) where {T<:Real}
    nt = size(X, 2)
    M = CuArray( diagm([i => ones(T, nt-1-abs(i)) for i in -pwin.Nindctr:pwin.Nindctr]...) )
    return M
end

function ar1_whitenoise(
    X::CuArray{T, 2},
    M::CuArray{T, 2},
) where {T<:Real}
    return ( (X[:, 2:end] .* X[:, 1:end-1]) * M ) ./ ( (X[:, 2:end] .^ 2) * M )
end

function ar1_whitenoise(X::CuArray{T, 3}, pwin::WindowingParams) where {T<:Real}
    TI = zeros(T, size(X,1), size(X,2)-1, size(X,3))
    for i in axes(TI, 1)
        TI[i, :, :] = Array( ar1_whitenoise( permutedims(X[i, :, :]), pwin ) )'
    end
    return TI
end

function ar1_whitenoise(X::Matrix{T}, pwin) where {T<:Real}
    nt = size(X, 2)
    M = spdiagm([i => ones(T, nt-1-abs(i)) for i in -pwin.Nindctr:pwin.Nindctr]...)
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

# Compute low-frequency power spectrum.
function lfps(x::Vector{T}; q_lowfreq=0.1) where {T<:Real}
    Psymmetrical = abs.(rfft(x) .^ 2)
    N = length(Psymmetrical)
    P = Psymmetrical[ roundint(N/2+1):end ]
    n = length(P)
    Pnorm = P ./ sum(P)
    nlow = roundint( n * q_lowfreq )
    return sum( Pnorm[1:nlow] )
end

function lfps(X::CuArray{T, 2}) where {T<:Real}
    P = abs.(CUDA.CUFFT.rfft( X, 2 )).^2
    Pnorm = P ./ reduce(+, P, dims=2)
    return reduce(+, Pnorm[:, 1:roundint(0.1 * size(Pnorm, 2))], dims=2)
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
# - if matrix input, apply mapslices.
function slide_estimator(x::Vector{T}, hw::Int, estimator) where {T<:Real}
    nx = length(x)
    stat = fill(T(NaN), nx)
    for i in (hw+1):(nx-hw)
        x_windowed = centered_window( x, i, hw )
        stat[i] = estimator(x_windowed)
    end
    return stat[ .!isnan.(stat) ]
end

function slide_estimator(x::Vector{T}, hw::Int, estimator, windowing::String) where {T<:Real}
    nx = length(x)
    stat = fill(T(NaN), nx)
    if windowing == "center"
        for i in (hw+1):(nx-hw)
            x_windowed = centered_window( x, i, hw )
            stat[i] = estimator(x_windowed)
        end
    elseif windowing == "left"
        for i in (2*hw+1):nx
            x_windowed = left_window( x, i, hw )
            stat[i] = estimator(x_windowed)
        end
    elseif windowing == "right"
        for i in 1:(nx-2*hw)
            x_windowed = right_window( x, i, hw )
            stat[i] = estimator(x_windowed)
        end
    end
    return stat
end

function slide_estimator(X::Union{Matrix{T}, Adjoint{T, Matrix{T}}}, hw::Int, estimator) where {T<:Real}
    return mapslices( x -> slide_estimator(x, hw, estimator), X, dims = 2 )
end

function slide_estimator(X::CuArray{T, 2}, pwin::WindowingParams, estimator) where {T<:Real}
    
    Nstride = pwin.Nstride
    Nindctr = pwin.Nindctr

    stride_idx = Nindctr+1:2*Nstride:size(X,2)-Nindctr
    TI = zeros(T, size(X, 1), length(stride_idx) )
    for j1 in eachindex(stride_idx)         # TODO this might be optimized with loop vectorization?
        j2 = stride_idx[j1]
        TI[:, j1] = Array( estimator( X[:, j2-Nindctr:j2+Nindctr] ) )
    end
    return TI
end

function slide_estimator(X::Array{T, 3}, hw::Int, estimator) where {T<:Real}
    TI = similar(X)
    for i in axes(TI, 1)
        TI[i, :, :] = slide_estimator( X[i, :, :]', hw, estimator )'
    end
    return TI
end