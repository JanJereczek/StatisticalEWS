using ImageFiltering, CairoMakie

t = 0:0.1:2*π
x = sin.(t)
y = x + 0.1 * ( rand(length(t)) .- 0.5 )

z = imfilter( y, Kernel.gaussian((3,)) )

fig = Figure()
ax  = Axis(fig[1,1])

lines!(ax, t, y)
lines!(ax, t, z)
fig