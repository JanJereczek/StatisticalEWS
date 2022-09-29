#####################################################
#%% Models from Boettner and Boers 2021
#####################################################

struct ModelParams
    θ::Real
    σθ::Real
    ρ::Real
    σρ::Real
end

function f_linear(du, u, p, t)
    du[1] = p.θ * u[1] + u[2]
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
    du[1] = p.σθ
    du[2] = 0.0
end

function g_ar1(du, u, p, t)
    du[1] = 0.0
    du[2] = p.σρ
end

rdw(t::AbstractFloat) = -1. + 2. * t