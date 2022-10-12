
function f_linear(u, p, t)
    return p.φ * u
end

function f_doublewell(u, p, t)
    return -u^3 + u + rdw(t)
end

function f_linear_ar1(u, p, t)
    du = similar(u)
    du[1] = p.φ * u[1] + u[2]
    du[2] = p.ρ * u[2]
    return du
end

function g_ar1(u, p, t)
    du = similar(u)
    du[1] = 0.0
    du[2] = p.σρ
    return du
end

function g_ar1(u, p, t)
    return 0.0
end

function ar1_noise(u, p, t)
    return p.ρ * u + white_noise(p.σρ)
end

ar1_noise = OrnsteinUhlenbeckProcess(
    pmodel.ρ,
    0.,
    pmodel.σρ,
    tspan[1],
    0.0,
)

function window_ar1(x::Vector{T}, hw::Int) where {T<:Real}
    nx = length(x)
    φ_vec = fill(NaN, nx)
    for i in (hw+1):(nx-hw)
        x_windowed = centered_window( x, i, hw )
        φ_vec[i] = get_ar1(x_windowed)
    end
    return φ_vec
end

function window_var(x::Vector{T}, hw::Int) where {T<:Real}
    nx = length(x)
    σ_vec = fill(NaN, nx)
    for i in (hw+1):(nx-hw)
        x_windowed = centered_window( x, i, hw )
        σ_vec[i] = std(x_windowed)
    end
    return σ_vec
end

###############################################################################

# Function to compute AR1 coefficient of a vector x for a given window size
function window_ar1(x, i1, i2, wsize)
    n = i2-i1
    ar1 = zeros(n)
    irange = i1:i2
    for j in 1:n
        i = irange[j]
        xwin = window(x, i, wsize)
        ar1[j] = get_ar1(xwin)
    end
    return ar1
end

φ = exp(-λ*Δt)
σtilde2 = 1 / (2*λ) * σ^2 * ( 1-exp(-2 * λ * Δt) )
σtilde = sqrt( σtilde2 )


# Compute AR1 coefficient θ of a vector x for AR1 noise assumption.
function ar1_ar1noise(x::Vector{T}) where {T<:Real}
    N = length(x)
    θb = ar1_whitenoise(x)
    V1 = x[3:N] - θb .* x[2:N-1]
    V0 = x[2:N-1] - θb .* x[1:N-2]
    ρb = V1' * V0 / ( V0' * V0 )
    a = θb + ρb
    b = ρb / θb
    Δ = a^2 - 4*b
    
    # if Δ < 0
    #     return NaN
    # elseif (1 > θb > ρb > -1)
    #     return (a + sqrt(Δ) ) / 2
    # elseif (1 > ρb > θb > -1)
    #     return (a - sqrt(Δ) ) / 2
    # else
    #     return NaN
    # end

    if (θb > ρb)
        return real( (a + sqrt( complex(Δ)) ) / 2 )
    elseif (ρb > θb)
        return real( (a - sqrt( complex(Δ)) ) / 2 )
    else
        return NaN
    end
end