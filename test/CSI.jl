using DifferentialEquations
using ControlSystemIdentification, ControlSystemsBase, Random, LinearAlgebra
using ControlSystemIdentification: newpem

function f_linear(du, u, p, t)
    du[1] = p.θ * u[1]
end

function g_whitenoise(du, u, p, t)
    du[1] = p.σθ
end

dt = 0.001
t = collect(0:dt:10)
tspan = extrema(t)
θ, ρ = -0.8, -0.1       # coefficients of AR1 models in continuous time domain.
σθ, σρ = 0.1, 1.        # standard deviation of noise processes.
pmodel = ModelParams(θ, σθ, ρ, σρ)
u0 = [0.0]
prob = SDEProblem(f_linear, g_whitenoise, u0, tspan, pmodel)

saved_values = SavedValues(Float64, Vector{Float64})
cb = SavingCallback((u,t,integrator)-> get_du(integrator), saved_values)
sol = solve(prob, EM(), dt=dt, save_noise = true, callback = cb)

u = vcat(sol.u...)
ducb = vcat(saved_values.saveval...)

fig = Figure(resolution=(1000, 1000))
axs = [Axis(fig[i, j]) for i in 1:2, j in 1:2]
lines!(axs[1], ducb)
lines!(axs[2], autocor(ducb))

# A = [-1.]
# B = [1.]
# C = [1.]
# D = [0.]
# guess_sys0 = ss(A,B,C,D)

d = iddata(u[2:end], ducb[1:end-1], dt)
sysh, opt = newpem(d, 1, focus=:prediction, iterations = 1000) # Estimate model
yh = predict(sysh, d) # Predict using estimated model
lines!(axs[3], vec(yh))
lines!(axs[3], u)
fig




###### from the doc

# sys = c2d(tf(1, [1, 0.5, 1]) * tf(1, [1, 1]), 0.1)
# Random.seed!(1)
# T   = 1000                      # Number of time steps
# nx  = 3                         # Number of poles in the true system
# nu  = 1                         # Number of inputs
# x0  = randn(nx)                 # Initial state
# sim(sys,u,x0=x0) = lsim(ss(sys), u, x0=x0).y # Helper function
# u   = randn(nu,T)               # Generate random input
# y   = sim(sys, u, x0)           # Simulate system
# y .+= 0.01 .* randn.()          # Add some measurement noise
# d   = iddata(y,u,0.1)
# sysh,opt = newpem(d, nx, focus=:prediction) # Estimate model
# yh = predict(sysh, d) # Predict using estimated model