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