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
    Twin::Real
    Tstep::Real
    Tind::Real
    T1::Real
    Nwin::Int
    Nstep::Int
    Nind::Int
    N1::Int
end

function roundint(x::Real)
    return Int( round( x ) )
end

function get_step(T::Real, T0::Real)
    return roundint(T / T0)
end