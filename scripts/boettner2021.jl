using DrWatson
@quickactivate "StatisticalEWS"
include(srcdir("init.jl"))

dt = 0.1
t = collect(0:dt:10)
tspan = extrema(t)
θ, ρ = -0.5, -0.1       # coefficients of AR1 models in continuous time domain.
σθ, σρ = 0.1, 1.        # standard deviation of noise processes.
pmodel = ModelParams(θ, σθ, ρ, σρ)
u0 = [-1.0, 0.0]

models = [f_linear, f_doublewell]
noises = [g_whitenoise, g_ar1]
nmodels = length(models)
nnoises = length(noises)
nrows = 4
ncols = nmodels * nnoises
ns = 100                # number of Fourier surrogates.

fig = Figure(resolution=(2000,800))
axs = [[Axis(fig[j,i]) for i in 1:ncols] for j in 1:nrows]
U = zeros(4, Int((tspan[2]-tspan[1])/dt + 1) )
Utrend, R = copy(U), copy(U)
Θwn, Θar1 = copy(U), copy(U)
Σ, LFPS = copy(U), copy(U)

Twin = 0.1      # window half-width.
Tstep = 0.02    # padding step.
Tind = 0.5      # window half-width for EWI computation.
T1 = dt
T = [dt, Twin, Tstep, Tind, T1]
N = get_step.(T[2:end], dt)
pwin = WindowingParams(T..., N...)

slide_ar1wn(x) = slide_estimator(x, pwin.Nwin, ar1_whitenoise)
slide_ar1cn(x) = slide_estimator(x, pwin.Nind, ar1_ar1noise)
slide_std(x) = slide_estimator(x, pwin.Nind, std)
slide_lfps(x) = slide_estimator(x, pwin.Nind, lfps)

for i in 1:2
    for j in 1:2
        Random.seed!(1995)
        f = models[i]
        g = noises[j]
        k = (i-1)*2+j

        prob = SDEProblem(f, g, u0, tspan, pmodel)
        sol = solve(prob, EM(), dt=dt)
        u = hcat(sol.u...)
        
        U[k, :] = u[1, :]
        Utrend[k, :] = gettrend_rollmean(u[1, :], pwin.Nwin)
        R[k, :] = detrend(U[k, :], Utrend[k, :])

        S = fourier_surrogates(R[k, :], ns)

        Θwn[k, :] = slide_estimator(R[k, :], pwin.Nwin, ar1_whitenoise)
        Θar1[k, :] = slide_estimator(R[k, :], pwin.Nind, ar1_ar1noise)
        Σ[k, :] = slide_estimator(R[k, :], pwin.Nind, std)
        LFPS[k, :] = slide_estimator(R[k, :], pwin.Nind, lfps)
        
        lines!(axs[1][k], sol.t, u[1, :], label = L"$u_1$")
        lines!(axs[1][k], sol.t, u[2, :], label = L"$u_2$")
        lines!(axs[1][k], sol.t, Utrend[k, :], linestyle = :dash, color = :black, label = L"\bar{u}_{1}")
        
        lines!(axs[2][k], sol.t, R[k, :], label = L"Residual $\,$")
        lines!(axs[2][k], sol.t, fourier_surrogate(R[k, :]), label = L"Fourier surrogate $\,$")

        lines!(axs[3][k], sol.t, Θwn[k, :], label = L"$\varphi_\mathrm{wn}$")
        lines!(axs[3][k], sol.t, Θar1[k, :], label = L"$\varphi_\mathrm{AR1}$")

        lines!(axs[4][k], sol.t, Σ[k, :], label = L"$\sigma$")
        lines!(axs[4][k], sol.t, LFPS[k, :], label = L"LFPS $\,$")
        
        if (k == ncols)
            for l in 1:nrows
                fig[l, ncols+1] = Legend(fig, axs[l][ncols], framevisible = false)
            end
        end
    end
end

fig

# slide_ar1wn(x) = slide_estimator(x, pwin.Nwin, ar1_whitenoise);
# pvalue = kendall_tau(R[1,:], S, slide_ar1wn)