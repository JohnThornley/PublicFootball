module PyPlotMeansFromProbresults
# ==============================================================================
# Plot grid of (meanA vs [Pr(winA), Pr(winB)]) and (meanB vs [Pr(winA), Pr(winB)]) plots
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

PROB_DIVISOR = 20
@assert PROB_DIVISOR % 2 == 0

MAX_MEAN = 4
MEAN_DIVISOR = 4
@assert 1 <= MAX_MEAN
@assert 1 <= MEAN_DIVISOR

CAMERA_ELEV = 30
CAMERA_AZIM = 210

MEAN_TOLERANCE = 1e-04

# ==============================================================================
# Derived fixed values for this plot
# ==============================================================================

PROB_RANGE = [0:1:PROB_DIVISOR;]/PROB_DIVISOR
MEAN_RANGE = [0:1:MAX_MEAN;]
MEAN_LEVELS = Float64.(MEAN_RANGE[2:length(MEAN_RANGE)])

# ==============================================================================
# Plotting
# ==============================================================================

function subplot!(ax, pointslist::Vector{Points3D}, strresult::LaTeXString, strcorrAB::LaTeXString, linewidth::Int64, plotfrontaxis::Bool)::Nothing
    zs = Iterators.flatten(points.zs for points in pointslist)
    @assert 0.0 <= minimum(zs)
    @assert maximum(zs) <= MAX_MEAN
    @assert 0 < length(strresult)
    @assert 0 < length(strcorrAB)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0:0.2:1.0;])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.0:0.2:1.0;])
    ax.set_zlim(0, MAX_MEAN)
    ax.set_zticks([0:1:MAX_MEAN;])
    ax.tick_params(axis="x", which="major", labelsize=10)
    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.view_init(azim=CAMERA_AZIM, elev=CAMERA_ELEV)
    ax.set_title(strresult*", "*L"$\rho_{A,B}=$"*strcorrAB)
    ax.set_xlabel(L"$P(win_{A})$")
    ax.set_ylabel(L"$P(win_{B})$")
    for points in pointslist
        ax.plot(points.xs, points.ys, zeros(length(points.xs)), color="lightgrey", linewidth=1, linestyle="solid", zorder=1)
    end
    for points in pointslist
        ax.plot(points.xs, points.ys, points.zs, color="blue", linewidth=linewidth, linestyle="solid", zorder=2)
    end
    ax.plot([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [MAX_MEAN, MAX_MEAN, MAX_MEAN], color="darkgrey", linewidth=1, linestyle="solid")
    if plotfrontaxis ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, MAX_MEAN], color="darkgrey", linewidth=1, linestyle="solid") end
    return
end

function subplots!(axmeanA, axmeanB, corrAB::Float64, strcorrAB::LaTeXString, linewidth::Int64, plotfrontaxis::Bool)::Nothing
    @assert 0 <= corrAB <= 1.0
    @assert 0 < length(strcorrAB)

    meanA::Array{Float64, 2} = fill(NaN, length(PROB_RANGE), length(PROB_RANGE))
    meanB::Array{Float64, 2} = fill(NaN, length(PROB_RANGE), length(PROB_RANGE))
    for (i, probwinA) in enumerate(PROB_RANGE)
        for (j, probwinB) in enumerate(PROB_RANGE)
            if probwinA + probwinB == 0.0 || corrAB < 1.0 && probwinA + probwinB < 1.0
                meanA[i, j], meanB[i, j] = means_from_probresults(probwinB, probwinA, corrAB, MEAN_TOLERANCE)
    end end end

    pointslistmeanA::Vector{Points3D} = []
    pointslistmeanB::Vector{Points3D} = []

    horizontalLevelsmeanA::Vector{Points3D} = [Points3D([], [], []) for level in 1:length(MEAN_LEVELS)]
    horizontalLevelsmeanB::Vector{Points3D} = [Points3D([], [], []) for level in 1:length(MEAN_LEVELS)]

    for (k, probwin) in enumerate(PROB_RANGE)
        println("k = $(k), probwin = $(probwin)")

        # Lines for fixed Pr(winA), varying Pr(winB) on meanA plot
        validj::Vector{Bool} = meanA[k, :] .< MAX_MEAN
        bs::Vector{Float64} = PROB_RANGE[validj]
        as::Vector{Float64} = fill(PROB_RANGE[k], length(bs))
        fixedwinAlinemeanA = Points3D(as, bs, meanA[k, validj])
        if (k == 1 && corrAB == 0.0) push!(fixedwinAlinemeanA, 0.0, 1.0, 0.0) end # add corner

        # Lines for fixed Pr(winA), varying Pr(winB) on meanB plot
        validj = meanB[k, :] .< MAX_MEAN
        bs = PROB_RANGE[validj]
        as = fill(PROB_RANGE[k], length(bs))
        fixedwinAlinemeanB = Points3D(as, bs, meanB[k, validj])

        # Lines for fixed Pr(winB), varying Pr(winA) on meanA plot
        validi::Vector{Bool} = meanA[:,k] .< MAX_MEAN
        as = PROB_RANGE[validi]
        bs = fill(PROB_RANGE[k], length(as))
        fixedwinBlinemeanA = Points3D(as, bs, meanA[validi, k])

        # Lines for fixed Pr(winB), varying Pr(winA) on meanB plot
        validi = meanB[:,k] .< MAX_MEAN
        as = PROB_RANGE[validi]
        bs = fill(PROB_RANGE[k], length(as))
        fixedwinBlinemeanB = Points3D(as, bs, meanB[validi, k])
        if (k == 1 && corrAB == 0.0) push!(fixedwinBlinemeanB, 1.0, 0.0, 0.0) end # add corner

        # Append to horizontal lines for each MEAN_LEVEL
        for (level, levelmean) in enumerate(MEAN_LEVELS)
            println("level = $(level), levelmean = $(levelmean)")
            println("proby1gty2_from_proby1lty2andmean1(probwin = $(probwin), levelmean = $(levelmean), corrAB = $(corrAB), MEAN_TOLERANCE)")
            probwinAFromB =
                if (probwin == 1.0)
                    if (corrAB == 0.0) 0.0 else NaN end
                else
                    proby1gty2_from_proby1lty2andmean1(probwin, levelmean, corrAB, MEAN_TOLERANCE)
                end
            if (!isnan(probwinAFromB))
                push!(fixedwinBlinemeanA, probwinAFromB, probwin, levelmean)
                push!(fixedwinAlinemeanB, probwin, probwinAFromB, levelmean)
                push!(horizontalLevelsmeanA[level], probwinAFromB, probwin, levelmean)
                push!(horizontalLevelsmeanB[level], probwin, probwinAFromB, levelmean)
            end
            println("proby1lty2_from_proby1gty2andmean1(probwin = $(probwin), levelmean = $(levelmean), corrAB = $(corrAB), MEAN_TOLERANCE)")
            probwinBFromA =
                if (probwin == 1.0)
                    NaN
                elseif (probwin == 0.0 && corrAB == 0.0)
                    1.0
                else
                    proby1lty2_from_proby1gty2andmean1(probwin, levelmean, corrAB, MEAN_TOLERANCE)
                end
            if !isnan(probwinBFromA)
                push!(fixedwinAlinemeanA, probwin, probwinBFromA, levelmean)
                push!(fixedwinBlinemeanB, probwinBFromA, probwin, levelmean)
                push!(horizontalLevelsmeanA[level], probwin, probwinBFromA, levelmean)
                push!(horizontalLevelsmeanB[level], probwinBFromA, probwin, levelmean)
            end
        end

        # Sort and append grid lines
        push!(pointslistmeanA, sorted(fixedwinAlinemeanA, fixedwinAlinemeanA.ys))
        push!(pointslistmeanB, sorted(fixedwinAlinemeanB, fixedwinAlinemeanB.ys))
        push!(pointslistmeanA, sorted(fixedwinBlinemeanA, fixedwinBlinemeanA.xs))
        push!(pointslistmeanB, sorted(fixedwinBlinemeanB, fixedwinBlinemeanB.xs))
    end

    # Sort and append horizontal lines for each MEAN_RANGE
    for level in 1:length(MEAN_LEVELS)
        push!(pointslistmeanA, sorted(horizontalLevelsmeanA[level], (1.0 .- horizontalLevelsmeanA[level].ys)./(1.0 .- horizontalLevelsmeanA[level].xs)))
        push!(pointslistmeanB, sorted(horizontalLevelsmeanB[level], (1.0 .- horizontalLevelsmeanB[level].xs)./(1.0 .- horizontalLevelsmeanB[level].ys)))
    end

    subplot!(axmeanA, pointslistmeanA, L"$\mu_{A}$", strcorrAB, linewidth, plotfrontaxis)
    subplot!(axmeanB, pointslistmeanB, L"$\mu_{B}$", strcorrAB, linewidth, plotfrontaxis)
    return
end

# ==============================================================================
# Main program
# ==============================================================================

fig = plt.figure(figsize=(8, 16))  # 2 columns, 5 nrows

nrows = length(CORRELATIONS)
ncols = 2
for row = 1:nrows
    corr = CORRELATIONS[row][1]
    strcorr = CORRELATIONS[row][2]
    axmeanA = fig.add_subplot(nrows, ncols, (row - 1)*ncols + 1, projection="3d")
    axmeanB = fig.add_subplot(nrows, ncols, (row - 1)*ncols + 2, projection="3d")
    subplots!(axmeanA, axmeanB, corr, strcorr, if row < nrows 1 else 2 end, if row < nrows true else false end)
end

fig.tight_layout(pad=2.5, h_pad=5.0)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\MeansFromProbresults.png")

# ==============================================================================
end
