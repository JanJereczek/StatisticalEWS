#####################################################
#%% Windowing
#####################################################
struct WindowingParams
    T0::Real
    T_smooth_wndw::Real
    T_indctr_wndw::Real
    T_indctr_strd::Real
    T_signif_wndw::Real
    T_signif_strd::Real
    N_smooth_wndw::Int
    N_indctr_wndw::Int
    N_indctr_strd::Int
    N_signif_wndw::Int
    N_signif_strd::Int
end

function get_windowing_params(Tvec::Vector{T}) where {T<:Real}
    N = get_step.(Tvec[2:end], dt)
    return WindowingParams(Tvec..., N...)
end

# Get window (half width hw) of vector x at index idx.
centered_window(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ (idx - hw):( idx + hw ) ]
left_window(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ (idx - 2*hw):idx ]
right_window(x::Vector{T}, idx::Int, hw::Int) where {T<:Real} = x[ idx:(idx + 2*hw) ]

# Get window (half width hw) of matrix x at index idx.
centered_window(X::Matrix{T}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - hw):( idx + hw ) ]
left_window(X::Matrix{T}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - 2*hw):idx ]
right_window(X::Matrix{T}, idx::Int, hw::Int) where {T<:Real} = X[ :, idx:(idx + 2*hw) ]

# Get window (half width hw) of matrix x at index idx.
centered_window(X::CuArray{T, 2}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - hw):( idx + hw ) ]
left_window(X::CuArray{T, 2}, idx::Int, hw::Int) where {T<:Real} = X[ :, (idx - 2*hw):idx ]
right_window(X::CuArray{T, 2}, idx::Int, hw::Int) where {T<:Real} = X[ :, idx:(idx + 2*hw) ]

trim_win(x::Vector{T}, hw::Int) where {T<:Real} = x[hw+1:end-hw]
trim_win(X::Matrix{T}, hw::Int) where {T<:Real} = X[:, hw+1:end-hw]

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

function get_step(T::Real, T0::Real)
    return roundint(T / T0)
end

function lines_numeric!(ax, t, x, lbl)
    filt = .!isnan.(x)
    x_filt = x[ filt ]
    t_filt = t[ filt ]
    lines!(ax, t_filt, x_filt, label = lbl)
end