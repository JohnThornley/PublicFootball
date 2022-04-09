module PyPlotProbresultsFromMeans
# ==============================================================================
# Plot a grid of [Pr(winA), Pr(draw), Pr(winB)] vs (meanA, meanB) plots
# for different corrAB
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; meanA, meanB, corrAB).
# ==============================================================================

import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot3d!(plt)
using LaTeXStrings  # needed for L notation

using Util
using BivariatePoisson

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

CORRELATIONS = [
    (0.0, L"0"),
    (sqrt(1/4), L"\sqrt{\frac{1}{4}}"),
    (sqrt(1/2), L"\sqrt{\frac{1}{2}}"),
    (sqrt(7/8), L"\sqrt{\frac{7}{8}}"),
    (1.0,  L"1")
]

MAX_MEAN = 4
MEAN_DIVISOR = 4
@assert 1 <= MAX_MEAN
@assert 1 <= MEAN_DIVISOR

CAMERA_ELEV = 30
CAMERA_AZIM = 35

# ==============================================================================
# Derived fixed values for this plot
# ==============================================================================

MEAN_RANGE = [0:1:(MAX_MEAN*MEAN_DIVISOR);]/MEAN_DIVISOR

# ==============================================================================
# Plotting
# ==============================================================================

function subplot!(ax, pointslist::Vector{Points3D}, strresult::LaTeXString, strcorrAB::LaTeXString, linewidth::Int64)::Nothing
    zs = Iterators.flatten(points.zs for points in pointslist)
    @assert 0.0 <= minimum(zs)
    @assert maximum(zs) <= 1.0
    @assert 0 < length(strresult)
    @assert 0 < length(strcorrAB)
    ax.set_xlim(0, MAX_MEAN)
    ax.set_xticks([0:1:MAX_MEAN;])
    ax.set_ylim(0, MAX_MEAN)
    ax.set_yticks([0:1:MAX_MEAN;])
    ax.set_zlim(0, 1.0)
    ax.set_zticks([0.0:0.2:1.0;])
    ax.view_init(azim=CAMERA_AZIM, elev=CAMERA_ELEV)
    ax.set_title(L"P(" *strresult* L"), \rho_{A,B}="*strcorrAB)
    ax.set_xlabel(L"$\mu_{A}$")
    ax.set_ylabel(L"$\mu_{B}$")
    for points in pointslist
        ax.plot(points.xs, points.ys, zeros(length(points.xs)), color="lightgrey", linewidth=1, linestyle="solid", zorder=1)
    end
    for points in pointslist
        ax.plot(points.xs, points.ys, points.zs, color="blue", linewidth=linewidth, linestyle="solid", zorder=2)
    end
    ax.plot([0.0, MAX_MEAN, MAX_MEAN], [MAX_MEAN, MAX_MEAN, 0.0], [1.0, 1.0, 1.0], color="darkgrey", linewidth=1, linestyle="solid", zorder=3)
    ax.plot([MAX_MEAN, MAX_MEAN], [MAX_MEAN, MAX_MEAN], [0.0, 1.0], color ="darkgrey", linewidth=1, linestyle="solid", zorder=3)
    return
end

function subplots!(axwinA, axdraw, axwinB, corrAB::Float64, strcorrAB::LaTeXString, linewidth::Int64)::Nothing
    @assert 0 <= corrAB <= 1.0
    @assert 0 < length(strcorrAB)

    probresults(meanA, meanB) = probresults_from_means(meanA, meanB, corrAB)

    probwinA::Array{Float64, 2} = fill(NaN, length(MEAN_RANGE), length(MEAN_RANGE))
    probwinB::Array{Float64, 2} = fill(NaN, length(MEAN_RANGE), length(MEAN_RANGE))
    for (i, meanA) in enumerate(MEAN_RANGE)
        for (j, meanB) in enumerate(MEAN_RANGE)
            if meansvalid(meanA, meanB, corrAB)
                probwinB[i, j], probwinA[i, j] = probresults(meanA, meanB)
    end end end
    probdraw::Array{Float64, 2} = 1.0 .- probwinA .- probwinB

    pointslistwinA::Vector{Points3D} = []
    pointslistdraw::Vector{Points3D} = []
    pointslistwinB::Vector{Points3D} = []

    frontedgewinA::Points3D = Points3D([0.0], [0.0], [probwinA[1, 1]])
    frontedgedraw::Points3D = Points3D([0.0], [0.0], [probdraw[1, 1]])
    frontedgewinB::Points3D = Points3D([0.0], [0.0], [probwinB[1, 1]])
    for (i, meanA) in enumerate(MEAN_RANGE)
        validj::Vector{Bool} = .!isnan.(probwinA[i, :])
        bs::Vector{Float64} = MEAN_RANGE[validj]
        as::Vector{Float64}  = fill(MEAN_RANGE[i], length(bs))
        pointswinA = Points3D(as, bs, probwinA[i, validj])
        pointsdraw = Points3D(as, bs, probdraw[i, validj])
        pointswinB = Points3D(as, bs, probwinB[i, validj])
        if 0.0 < meanA && 0.0 < corrAB
            minmeanB = minsmallermean(meanA, corrAB)
            maxmeanB = min(maxlargermean(meanA, corrAB), Float64(MAX_MEAN))
            pwinB, pwinA = probresults(meanA, minmeanB)
            pdraw = 1.0 - pwinA - pwinB
            pushfirst!(pointswinA, meanA, minmeanB, pwinA)
            pushfirst!(pointsdraw, meanA, minmeanB, pdraw)
            pushfirst!(pointswinB, meanA, minmeanB, pwinB)
            push!(frontedgewinA, meanA, minmeanB, pwinA)
            push!(frontedgedraw, meanA, minmeanB, pdraw)
            push!(frontedgewinB, meanA, minmeanB, pwinB)
            pwinB, pwinA = probresults(meanA, maxmeanB)
            pdraw = 1.0 - pwinA - pwinB
            push!(pointswinA, meanA, maxmeanB, pwinA)
            push!(pointsdraw, meanA, maxmeanB, pdraw)
            push!(pointswinB, meanA, maxmeanB, pwinB)
        end
        push!(pointslistwinA, pointswinA)
        push!(pointslistdraw, pointsdraw)
        push!(pointslistwinB, pointswinB)
    end
    if 0.0 < corrAB
        push!(pointslistwinA, frontedgewinA)
        push!(pointslistdraw, frontedgedraw)
        push!(pointslistwinB, frontedgewinB)
    end

    frontedgewinA = Points3D([0.0], [0.0], [probwinA[1, 1]])
    frontedgedraw = Points3D([0.0], [0.0], [probdraw[1, 1]])
    frontedgewinB = Points3D([0.0], [0.0], [probwinB[1, 1]])
    for (j, meanB) in enumerate(MEAN_RANGE)
        validi::Vector{Bool} = .!isnan.(probwinA[:, j])
        as::Vector{Float64} = MEAN_RANGE[validi]
        bs::Vector{Float64} = fill(MEAN_RANGE[j], length(as))
        pointswinA = Points3D(as, bs, probwinA[validi, j])
        pointsdraw = Points3D(as, bs, probdraw[validi, j])
        pointswinB = Points3D(as, bs, probwinB[validi, j])
        if 0.0 < meanB && 0.0 < corrAB
            minmeanA = minsmallermean(meanB, corrAB)
            maxmeanA = min(maxlargermean(meanB, corrAB), Float64(MAX_MEAN))
            pwinB, pwinA = probresults(minmeanA, meanB)
            pdraw = 1.0 - pwinA - pwinB
            pushfirst!(pointswinA, minmeanA, meanB, pwinA)
            pushfirst!(pointsdraw, minmeanA, meanB, pdraw)
            pushfirst!(pointswinB, minmeanA, meanB, pwinB)
            push!(frontedgewinA, minmeanA, meanB, pwinA)
            push!(frontedgedraw, minmeanA, meanB, pdraw)
            push!(frontedgewinB, minmeanA, meanB, pwinB)
            pwinB, pwinA = probresults(maxmeanA, meanB)
            pdraw = 1.0 - pwinA - pwinB
            push!(pointswinA, maxmeanA, meanB, pwinA)
            push!(pointsdraw, maxmeanA, meanB, pdraw)
            push!(pointswinB, maxmeanA, meanB, pwinB)
        end
        push!(pointslistwinA, pointswinA)
        push!(pointslistdraw, pointsdraw)
        push!(pointslistwinB, pointswinB)
    end
    if 0.0 < corrAB
        push!(pointslistwinA, frontedgewinA)
        push!(pointslistdraw, frontedgedraw)
        push!(pointslistwinB, frontedgewinB)
    end

    subplot!(axwinA, pointslistwinA, L"win_A", strcorrAB, linewidth)
    subplot!(axdraw, pointslistdraw, L"draw", strcorrAB, linewidth)
    subplot!(axwinB, pointslistwinB, L"win_B", strcorrAB, linewidth)
    return
end

# ==============================================================================
# Main program
# ==============================================================================

fig = plt.figure(figsize=(10, 16))  # 3 columns, 5 rows

nrows = length(CORRELATIONS)
ncols = 3
for row = 1:nrows
    corr = CORRELATIONS[row][1]
    strcorr = CORRELATIONS[row][2]
    axwinA = fig.add_subplot(nrows, ncols, (row - 1)*ncols + 1, projection="3d")
    axdraw = fig.add_subplot(nrows, ncols, (row - 1)*ncols + 2, projection="3d")
    axwinB = fig.add_subplot(nrows, ncols, (row - 1)*ncols + 3, projection="3d")
    subplots!(axwinA, axdraw, axwinB, corr, strcorr, if row < nrows 1 else 2 end)
end

fig.tight_layout(pad=2.5, h_pad=5.0)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\ProbresultsFromMeans.png")

# ==============================================================================
end
