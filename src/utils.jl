#####################################################
#%% Windowing
#####################################################
struct WindowingParams
    T0::Real
    T_smooth_win::Real
    T_indctr_win::Real
    T_indctr_stride::Real
    T_signif_win::Real
    T_signif_stride::Real
    N_smooth_win::Int
    N_indctr_win::Int
    N_indctr_stride::Int
    N_signif_win::Int
    N_signif_stride::Int
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