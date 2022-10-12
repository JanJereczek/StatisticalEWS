include(srcdir("init.jl"))

function main(random_seed, trend_measure)
    dt = 0.02
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
    nrows = 3
    ncols = nmodels * nnoises
    ns = 100                # number of Fourier surrogates.
    λ = 0.1                 # regularization parameter of ridge regression.

    fig = Figure(resolution=(2000,800))
    axs = [[Axis(fig[j,i]) for i in 1:ncols] for j in 1:nrows]
    U = zeros(4, Int((tspan[2]-tspan[1])/dt + 1) )
    Utrend, R = copy(U), copy(U)
    Θwn, Θcn = copy(U), copy(U)
    Var, LFPS = copy(U), copy(U)

    Tsmooth = 0.1       # half-width of smoothing window .
    Tindctr = 0.5       # half-width for computation of indicator.
    Tstride = 0.2       # stride for computation of indicator.
    T = [dt, Tsmooth, Tindctr, Tstride]
    N = get_step.(T[2:end], dt)
    pwin = WindowingParams(T..., N...)

    slide_ar1wn(x) = slide_estimator(x, pwin.Nindctr, ar1_whitenoise)
    slide_ar1cn(x) = slide_estimator(x, pwin.Nindctr, ar1_ar1noise)
    slide_var(x) = slide_estimator(x, pwin.Nindctr, variance)
    slide_lfps(x) = slide_estimator(x, pwin.Nindctr, lfps)

    for i in 1:2
        for j in 1:2
            Random.seed!(random_seed)
            f = models[i]
            g = noises[j]
            k = (i-1)*2+j

            prob = SDEProblem(f, g, u0, tspan, pmodel)
            sol = solve(prob, EM(), dt=dt)
            u = hcat(sol.u...)
            
            U[k, :] = u[1, :]
            Utrend[k, :] = gettrend_rollmean(u[1, :], pwin.Nsmooth)
            R[k, :] = detrend(U[k, :], Utrend[k, :])
            S = fourier_surrogates(R[k, :], ns)

            lines!(axs[1][k], sol.t, u[1, :], label = L"$u_1$")
            lines!(axs[1][k], sol.t, u[2, :], label = L"$u_2$")
            lines!(axs[1][k], sol.t, Utrend[k, :], linestyle = :dash, color = :black, label = L"\bar{u}_{1}")
            
            lines!(axs[2][k], sol.t, R[k, :], label = L"Residual $\,$")
            lines!(axs[2][k], sol.t, S[1, :], label = L"Fourier surrogate $\,$")

            EWI = Dict()

            for indicator in [ar1_whitenoise, ar1_ar1noise, variance, lfps]
                
                key = string(indicator)
                EWI[key] = Dict()

                EWI[key]["indctr_ref"] = slide_estimator(R[k, :], pwin.Nindctr, indicator)
                EWI[key]["indctr_srg"] = slide_estimator(S, pwin.Nindctr, indicator)
                
                if trend_measure == "kendall-tau"
                    EWI[key]["trend_ref"] = slide_kendall_tau(t, EWI[key]["indctr_ref"], pwin.Nindctr, pwin.Nstride)
                    EWI[key]["trend_srg"] = slide_kendall_tau(t, EWI[key]["indctr_srg"], pwin.Nindctr, pwin.Nstride)
                elseif trend_measure == "regression"
                    EWI[key]["trend_ref"] = slide_regression(t, EWI[key]["indctr_ref"], pwin.Nindctr, λ, pwin.Nstride)
                    EWI[key]["trend_srg"] = slide_regression(t, EWI[key]["indctr_srg"], pwin.Nindctr, λ, pwin.Nstride)
                end

                EWI[key]["p"] = percentile_significance(EWI[key]["trend_ref"], EWI[key]["trend_srg"])
                wsave(datadir("Boettner2021/", string(random_seed, "_", f, "_", g, "_", trend_measure, ".jld2")), EWI)
                println(size(EWI[key]["p"]))
                lines_numeric!(axs[3][k], t, EWI[key]["p"], key)
            end

            # Θwn[k, :] = slide_estimator(R[k, :], pwin.Nindctr, ar1_whitenoise)
            # Θcn[k, :] = slide_estimator(R[k, :], pwin.Nindctr, ar1_ar1noise)
            # Var[k, :] = slide_estimator(R[k, :], pwin.Nindctr, variance)
            # LFPS[k, :] = slide_estimator(R[k, :], pwin.Nindctr, lfps)
            
            # Θwn_surrogate = slide_estimator(S, pwin.Nindctr, ar1_whitenoise)
            # Θcn_surrogate = slide_estimator(S, pwin.Nindctr, ar1_ar1noise)
            # Var_surrogate = slide_estimator(S, pwin.Nindctr, variance)
            # LFPS_surrogate = slide_estimator(S, pwin.Nindctr, lfps)

            # lines!(axs[3][k], sol.t, Θwn[k, :], label = L"$\varphi_\mathrm{wn}$")
            # lines!(axs[3][k], sol.t, Θcn[k, :], label = L"$\varphi_\mathrm{AR1}$")

            # lines!(axs[4][k], sol.t, Var[k, :], label = L"$\sigma$")
            # lines!(axs[4][k], sol.t, LFPS[k, :], label = L"LFPS $\,$")
            
            if (k == ncols)
                for l in 1:nrows
                    fig[l, ncols+1] = Legend(fig, axs[l][ncols], framevisible = false)
                end
            end
        end
    end
    save_fig( plotsdir("Boettner2021/"), string(trend_measure, "_", random_seed), "both", fig)
end

function makesim(dicts)
    for (i, d) in enumerate(dicts)
        @unpack random_seed, trend_measure = d
        main(random_seed, trend_measure)
    end
end

random_seed = [1995, 2019, 2049]
trend_measure = ["kendall-tau", "regression"]
params = @strdict random_seed trend_measure
dicts = dict_list(params)
makesim(dicts)