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

function kendall_tau(x::Vector{T}, ns::Int, percentile_threshold::Real) where {T<:Real}
    nx = length(x)
    stat = zeros(ns)
    S = fourier_surrogates(x, ns)

    for i in axes(S, 1)
        stat[i] = corkendall(S[i,:], S[i,:])
    end
    # p = 1 - percentile(stat, percentile_threshold)/100
    # return p
    return stat
end

function slide_kendall_tau(x::Vector{T}, ns::Int, percentile_threshold::Real) where {T<:Real}
end

function quantile_estimation()
end