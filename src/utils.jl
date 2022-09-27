# Save a figure in pdf, png or both.
function save_fig(prefix, filename, extension, fig)
    if (extension == "pdf") || (extension == "both")
        save(string(prefix, filename, ".pdf"), fig)
    end
    if (extension == "png") || (extension == "both")
        save(string(prefix, filename, ".png"), fig)
    end
end

struct Params
    T0::Real
    Twin::Real
    Tstep::Real
    Tind::Real
    T1::Real
    Rwin::Int
    Rstep::Int
    Rind::Int
    R1::Int
end

function get_step(T::Real, T0::Real)
    return Int( round(T / T0) )
end