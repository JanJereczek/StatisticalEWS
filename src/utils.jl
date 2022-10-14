using LinearAlgebra, SparseArrays, CairoMakie, CUDA

#####################################################
#%% Windowing
#####################################################
# struct WindowingParams
#     T0::Real
#     T_smooth_wndw::Real
#     T_indctr_wndw::Real
#     T_indctr_strd::Real
#     T_signif_wndw::Real
#     T_signif_strd::Real
#     N_smooth_wndw::Int
#     N_indctr_wndw::Int
#     N_indctr_strd::Int
#     N_signif_wndw::Int
#     N_signif_strd::Int
# end

struct WindowingParams
    dt::Real
    Twndw::Real
    Tstrd::Real
    Nwndw::Int
    Nstrd::Int
end

function get_windowing_params(Tvec::Vector{T}) where {T<:Real}
    N = get_step.(Tvec[2:end], dt)
    return WindowingParams(Tvec..., N...)
end

# In the coming lines, some severe multiple dispatching is taking place.
# This is used for the user to simply specify the window function of their choice.
# All the rest happens under the hood.

# Get window (half width hw) of vector x at index idx.
centered_wndw(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ (idx - hw):( idx + hw ) ]
left_wndw(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ (idx - 2*hw):idx ]
right_wndw(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ idx:(idx + 2*hw) ]

# Get window (half width hw) of matrix x at index idx.
centered_wndw(X::Matrix{T}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - hw):( idx + hw ) ]
left_wndw(X::Matrix{T}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - 2*hw):idx ]
right_wndw(X::Matrix{T}, idx::Int, hw::Int) where {T<:Real} = X[ :, idx:(idx + 2*hw) ]

# Get window (half width hw) of matrix x at index idx.
centered_wndw(X::CuArray{T, 2}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - hw):( idx + hw ) ]
left_wndw(X::CuArray{T, 2}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - 2*hw):idx ]
right_wndw(X::CuArray{T, 2}, idx::Int, hw::Int) where {T<:Real} = X[ :, idx:(idx + 2*hw) ]

# Gives back the indices required for sliding as defined by the WindowingParams struct.
centered_wndw(n_wndw::Int, n_strd::Int, nt::Int) = (n_wndw+1):n_strd:(nt-n_wndw)
left_wndw(n_wndw::Int, n_strd::Int, nt::Int) = (2*n_wndw+1):n_strd:nt
right_wndw(n_wndw::Int, n_strd::Int, nt::Int) = 1:n_strd:(nt-2*n_wndw)

function trim_wndw(
    x::Vector{T},
    p::WindowingParams,
    wndw,
) where {T<:Real} 

    return x[ wndw(p.Nwndw, p.Nstrd, length(x)) ]
end

function trim_wndw(
    X::Matrix{T},
    p::WindowingParams,
    wndw,
) where {T<:Real} 

    return X[ :, wndw(p.Nwndw, p.Nstrd, size(X, 2)) ]
end

trim_wndw(x::Vector{T}, hw::Int) where {T<:Real} = x[hw+1:end-hw]
trim_wndw(X::Matrix{T}, hw::Int) where {T<:Real} = X[:, hw+1:end-hw]

trim_wndw(x::Vector{T}, n_wndw::Int, n_strd::Int, wndw) where {T<:Real} = x[wndw(n_wndw, n_strd, length(x))]

#####################################################
#%% Saving
#####################################################
# Save a figure in pdf, png or both.
function save_fig(prefix, filename, extension, fig)
    if (extension == "pdf") || (extension == "both")
        save(string(prefix, filename, ".pdf"), fig)
    end
    if (extension == "png") || (extension == "both")
        save(string(prefix, filename, ".png"), fig)
    end
end

#####################################################
#%% Misc
#####################################################
function roundint(x::Real)
    return Int( round( x ) )
end

function get_step(time::Real, dt::Real)
    return roundint(time / dt)
end

function lines_numeric!(ax, t, x, lbl)
    filt = .!isnan.(x)
    x_filt = x[ filt ]
    t_filt = t[ filt ]
    lines!(ax, t_filt, x_filt, label = lbl)
end