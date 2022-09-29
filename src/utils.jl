# Save a figure in pdf, png or both.
function save_fig(prefix, filename, extension, fig)
    if (extension == "pdf") || (extension == "both")
        save(string(prefix, filename, ".pdf"), fig)
    end
    if (extension == "png") || (extension == "both")
        save(string(prefix, filename, ".png"), fig)
    end
end

struct WindowingParams
    T0::Real
    Tsmooth::Real
    Tindctr::Real
    Tstride::Real
    Nsmooth::Int
    Nindctr::Int
    Nstride::Int
end

function roundint(x::Real)
    return Int( round( x ) )
end

function get_step(T::Real, T0::Real)
    return roundint(T / T0)
end

function lines_numeric!(ax, t, x, lbl)
    filt = .!isnan.(x)
    x_filt = x[ filt ]
    t_filt = t[ filt ]
    lines!(ax, t_filt, x_filt, label = lbl)
end