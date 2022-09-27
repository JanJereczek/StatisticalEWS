using DrWatson
@quickactivate "StatisticalEWS"

using Statistics, StatsBase, Random, Distributions
using CairoMakie, DifferentialEquations, FFTW
using BenchmarkTools, RollingFunctions

include(srcdir("ews.jl"))
include(srcdir("utils.jl"))
include(srcdir("models.jl"))
include(srcdir("surrogates.jl"))

dt = 0.01
t = collect(0:dt:10)
tspan = extrema(t)
pmodel = ModelParams(-1, .1, -.1, 1)
u0 = [-1.0, 0.0]

models = [f_linear, f_doublewell]
noises = [g_whitenoise, g_ar1]
nmodels = length(models)
nnoises = length(noises)
nrows = 2
ncols = nmodels * nnoises

fig = Figure(resolution=(2000,500))
axs = [[Axis(fig[j,i]) for i in 1:ncols] for j in 1:nrows]
u_dict = Dict()

for i in 1:2
    for j in 1:2
        Random.seed!(1995)
        f = models[i]
        g = noises[j]
        prob = SDEProblem(f, g, u0, tspan, pmodel)
        sol = solve(prob, EM(), dt=dt)
        u = hcat(sol.u...)
        u_dict[string(i, "_", j)] = u
        lines!(axs[1][(i-1)*2+j], sol.t, u[1, :])
        lines!(axs[1][(i-1)*2+j], sol.t, u[2, :])
    end
end

T = [dt, 0.1, 0.02, 0.02, dt]
R = get_step.(T[2:end], dt)
pstats = Params(T..., R...)

fig

# fig = Figure()
# axs = [Axis(fig[i,j]) for i in 1:2, j in 1:2]
# lines!(axs[1], t, y )
# lines!(axs[3], t, fourier_surrogate(y) )