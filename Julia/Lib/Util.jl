module Util

import Base: length, pushfirst!, push!

export Points3D, sorted

# ==============================================================================
# Points3D
# ==============================================================================

struct Points3D
    xs::Vector{Float64}
    ys::Vector{Float64}
    zs::Vector{Float64}

    function Points3D(xs, ys, zs)
        @assert length(xs) == length(ys) == length(zs)
        new(copy(xs), copy(ys), copy(zs))
    end
end

function length(points::Points3D)::Int32
    @assert length(points.xs) == length(points.ys) == length(points.zs)
    return length(points.xs)
end

function pushfirst!(points::Points3D, x::Float64, y::Float64, z::Float64)::Nothing
    @assert length(points.xs) == length(points.ys) == length(points.zs)
    prepend!(points.xs, x)
    prepend!(points.ys, y)
    prepend!(points.zs, z)
    return
end

function push!(points::Points3D, x::Float64, y::Float64, z::Float64)::Nothing
    @assert length(points.xs) == length(points.ys) == length(points.zs)
    append!(points.xs, x)
    append!(points.ys, y)
    append!(points.zs, z)
    return
end

function sorted(points::Points3D, keyvalues::Vector{Float64})::Points3D
    @assert length(keyvalues) == length(points.xs) == length(points.ys) == length(points.zs)
    perm = sortperm(keyvalues)
    return Points3D(points.xs[perm], points.ys[perm], points.zs[perm])
end

# ==============================================================================
end
