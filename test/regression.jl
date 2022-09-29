using DrWatson
@quickactivate "StatisticalEWS"
include(srcdir("init.jl"))

a, b = 1.3, 7.34
x = collect(-1:0.3:5.2)
y = a .* x .+ b
z = y + rand(Normal(0, 1), length(x))

w1 = linear_regression(x, y)
w2 = linear_regression(x, z)
w3 = ridge_regression(x, z, 0.1)