#####################################################
#%% Models from Boettner and Boers 2021
#####################################################

struct ModelParams
    φ::Real
    σφ::Real
    ρ::Real
    σρ::Real
end

function f_linear(du, u, p, t)
    du[1] = p.φ * u[1] + u[2]
    du[2] = p.ρ * u[2]
end

function f_doublewell(du, u, p, t)
    du[1] = -u[1]^3 + u[1] + u[2]
    du[2] = p.ρ * u[2]
end

function f_forceddoublewell(du, u, p, t)
    du[1] = -u[1]^3 + u[1] + rdw(t) + u[2]
    du[2] = p.ρ * u[2]
end

function g_whitenoise(du, u, p, t)
    du[1] = p.σφ
    du[2] = 0.0
end

function g_ar1(du, u, p, t)
    du[1] = 0.0
    du[2] = p.σρ
end

# function f_linear(u, p, t)
#     return p.φ * u
# end

# function f_doublewell(u, p, t)
#     return -u^3 + u + rdw(t)
# end

# function f_linear_ar1(u, p, t)
#     du = similar(u)
#     du[1] = p.φ * u[1] + u[2]
#     du[2] = p.ρ * u[2]
#     return du
# end

# function g_ar1(u, p, t)
#     du = similar(u)
#     du[1] = 0.0
#     du[2] = p.σρ
#     return du
# end

# function g_ar1(u, p, t)
#     return 0.0
# end

# function ar1_noise(u, p, t)
#     return p.ρ * u + white_noise(p.σρ)
# end

# ar1_noise = OrnsteinUhlenbeckProcess(
#     pmodel.ρ,
#     0.,
#     pmodel.σρ,
#     tspan[1],
#     0.0,
# )

rdw(t::AbstractFloat) = -1. + 2. * t