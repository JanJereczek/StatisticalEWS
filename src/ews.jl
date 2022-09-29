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

# Compute AR1 coefficient θ of a vector x for white noise assumption.
function ar1_whitenoise(x::Vector{T}) where {T<:Real}
    N = length(x)
    num = x[2:N]' * x[1:N-1]
    denum = x[2:N]' * x[2:N]
    θ = num / denum
    return θ
end

# Compute AR1 coefficient θ of a vector x for AR1 noise assumption.
function ar1_ar1noise(x::Vector{T}) where {T<:Real}
    N = length(x)
    θb = ar1_whitenoise(x)
    V1 = x[3:N] - θb .* x[2:N-1]
    V0 = x[2:N-1] - θb .* x[1:N-2]
    ρb = V1' * V0 / ( V0' * V0 )
    a = θb + ρb
    b = ρb / θb

    if (a^2 - 4*b) < 0
        return NaN
    elseif (1 > θb > ρb > -1)
        return (a + sqrt( a^2 - 4*b )) / 2
    elseif (1 > ρb > θb > -1)
        return (a - abs(a^2 - 4*b) ) / 2
    else
        return NaN
    end

end

# Compute low-frequency power spectrum.
function lfps(x::Vector{T}; q_lowfreq=0.1) where {T<:Real}
    Psymmetrical = abs.(fft(x) .^ 2)
    N = length(Psymmetrical)
    P = Psymmetrical[ roundint(N/2+1):end ]
    n = length(P)
    Pnorm = P ./ sum(P)
    nlow = roundint( n * q_lowfreq )
    return sum( Pnorm[1:nlow] )
end

# TODO implement EWSs below
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
