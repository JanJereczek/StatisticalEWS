#####################################################
#%% Windowing and Smoothing
#####################################################

# Get window (half width hw) of vector x at index idx.
centered_window(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ (idx - hw):( idx + hw ) ]
left_window(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ (idx - 2*hw):idx ]
right_window(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ idx:(idx + 2*hw) ]

function gettrend_rollmean(x::Vector{T}, hw::Int) where {T<:Real}
    xtrend = rollmean(x, 2*hw+1)
    xtrend = vcat( x[1:hw], xtrend, x[end-hw+1:end] )
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
#%% EWIs
#####################################################

# Compute AR1 coefficient ar1 of a vector x.
function get_ar1_whitenoise(x::Vector{T}) where {T<:Real}
    N = length(x)
    num = x[2:N]' * x[1:N-1]
    denum = x[2:N]' * x[2:N]
    φ = num / denum
    return φ
end

function get_ar1_ar1noise(x::Vector{T}) where {T<:Real}
    N = length(x)
    φb = get_ar1_whitenoise(x)
    V1 = x[3:N] - φb * y[2:N-1]
    V0 = x[2:N] - φb * y[1:N-1]
    ρb = V1' * V0 / ( V0' * V0 )
    a = φb + ρb
    b = ρb / φb

    if 1 > φb > ρb > -1
        return (a + sqrt(a-4*b)) / 2
    elseif 1 > ρb > φb > -1
        return (a - sqrt(a-4*b)) / 2
    else
        return NaN
    end
end

# TODO implement low-frequency power spectrum as EWS (and others below)
function lfps(x::Vector{T}, q_lowfreq::AbstractFloat) where {T<:Real}
    F = fft(fftshift(x))
    P = F .^ 2
    Pnorm = P ./ sum(P)
    nlow = Int( floor( n * q_lowfreq ) )
    return sum( Pnorm[1:nlow] )
end

function skewness()
end

function kurtosis()
end

function flickering()
end

function recovery_rate()
end

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

function return_rate()
end

function check_std_endpoint(x::Vector{T}, xwin::Vector{T}, std_tol::Real) where {T<:Real}
    return max(xwin) > std_tol * std(x)
end

#####################################################
#%% From EWIs to EWS
#####################################################

function slide_estimator(x::Vector{T}, hw::Int, estimator) where {T<:Real}
    nx = length(x)
    stat = fill(NaN, nx)
    for i in (hw+1):(nx-hw)
        x_windowed = centered_window( x, i, hw )
        stat[i] = estimator(x_windowed)
    end
    return stat
end

# function window_ar1(x::Vector{T}, hw::Int) where {T<:Real}
#     nx = length(x)
#     φ_vec = fill(NaN, nx)
#     for i in (hw+1):(nx-hw)
#         x_windowed = centered_window( x, i, hw )
#         φ_vec[i] = get_ar1(x_windowed)
#     end
#     return φ_vec
# end

# function window_var(x::Vector{T}, hw::Int) where {T<:Real}
#     nx = length(x)
#     σ_vec = fill(NaN, nx)
#     for i in (hw+1):(nx-hw)
#         x_windowed = centered_window( x, i, hw )
#         σ_vec[i] = std(x_windowed)
#     end
#     return σ_vec
# end

################################################################################

# Function to compute AR1 coefficient of a vector x for a given window size
# function window_ar1(x, i1, i2, wsize)
#     n = i2-i1
#     ar1 = zeros(n)
#     irange = i1:i2
#     for j in 1:n
#         i = irange[j]
#         xwin = window(x, i, wsize)
#         ar1[j] = get_ar1(xwin)
#     end
#     return ar1
# end

# φ = exp(-λ*Δt)
# σtilde2 = 1 / (2*λ) * σ^2 * ( 1-exp(-2 * λ * Δt) )
# σtilde = sqrt( σtilde2 )