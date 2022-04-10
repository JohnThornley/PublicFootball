module PlotProbscoreFromMeans
# ==============================================================================
# Plot a grid of Pr(goalsA, goalsB) vs (meanA, meanB) plots
# each for different goalsA, goalsB, and corrAB
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; meanA, meanB, corrAB).
# ==============================================================================

using Plots
using Plots.PlotMeasures  # needed for mm notation
using LaTeXStrings  # needed for L notation
using LinearAlgebra

using BivariatePoisson
using Util

pyplot()

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

MAX_MEAN = 4
MEAN_DIVISOR = 5
@assert 1 <= MAX_MEAN
@assert 1 <= MEAN_DIVISOR

CAMERA_ELEV = 30
CAMERA_AZIM = 125

# ==============================================================================
# Derived fixed values for this plot
# ==============================================================================

MEAN_RANGE = [0:1:(MAX_MEAN*MEAN_DIVISOR);]/MEAN_DIVISOR

# ==============================================================================
# Plotting
# ==============================================================================

function subplot(goalsA::Int64, goalsB::Int64, corrAB::Float64, strcorrAB::LaTeXString, linewidth::Int64)::Plots.Plot
    @assert 0 <= goalsA
    @assert 0 <= goalsB
    @assert 0 <= corrAB <= 1.0
    @assert 0 < length(strcorrAB)

    probgoalsAgoalsB(meanA, meanB) = probscore_from_means(goalsA, goalsB, meanA, meanB, corrAB)
    prob::Array{Float64, 2} = fill(NaN, length(MEAN_RANGE), length(MEAN_RANGE))
    for (i, meanA) in enumerate(MEAN_RANGE)
        for (j, meanB) in enumerate(MEAN_RANGE)
            if (i == j == 1) || meansvalid(meanA, meanB, corrAB)
                prob[i, j] =
                    if i == j == 1
                        if (goalsA == goalsB == 0) 1.0 else 0.0 end
                    else
                        probgoalsAgoalsB(meanA, meanB)
                    end
    end end end

    pointslist::Vector{Points3D} = []

    frontedge::Points3D = Points3D([0.0], [0.0], [prob[1, 1]])
    for (i, meanA) in enumerate(MEAN_RANGE)
        validj::Vector{Bool} = .!isnan.(prob[i, :])
        bs::Vector{Float64} = MEAN_RANGE[validj]
        as::Vector{Float64}  = fill(MEAN_RANGE[i], length(bs))
        points = Points3D(as, bs, prob[i, validj])
        if 0.0 < meanA && 0.0 < corrAB
            minmeanB = minsmallermean(meanA, corrAB)
            maxmeanB = min(maxlargermean(meanA, corrAB), Float64(MAX_MEAN))
            p = probgoalsAgoalsB(meanA, minmeanB)
            pushfirst!(points, meanA, minmeanB, p)
            push!(frontedge, meanA, minmeanB, p)
            p = probgoalsAgoalsB(meanA, maxmeanB)
            push!(points, meanA, maxmeanB, p)
        end
        push!(pointslist, points)
    end
    if 0.0 < corrAB
        push!(pointslist, frontedge)
    end

    frontedge = Points3D([0.0], [0.0], [prob[1, 1]])
    for (j, meanB) in enumerate(MEAN_RANGE)
        validi::Vector{Bool} = .!isnan.(prob[:, j])
        as::Vector{Float64} = MEAN_RANGE[validi]
        bs::Vector{Float64} = fill(MEAN_RANGE[j], length(as))
        points = Points3D(as, bs, prob[validi, j])
        if 0.0 < meanB && 0.0 < corrAB
            minmeanA = minsmallermean(meanB, corrAB)
            maxmeanA = min(maxlargermean(meanB, corrAB), Float64(MAX_MEAN))
            p = probgoalsAgoalsB(minmeanA, meanB)
            pushfirst!(points, minmeanA, meanB, p)
            push!(frontedge, minmeanA, meanB, p)
            p = probgoalsAgoalsB(maxmeanA, meanB)
            push!(points, maxmeanA, meanB, p)
        end
        push!(pointslist, points)
    end
    if 0.0 < corrAB
        push!(pointslist, frontedge)
    end

    zmax::Float64 = maximum([ceil(maximum(points.zs), digits = 1) for points in pointslist])

    plt = plot(xlim = (0, MAX_MEAN), ylim = (0, MAX_MEAN), zlim = (0.0, 1.0), legend = false)
    plot!(camera = (CAMERA_AZIM, CAMERA_ELEV))
    plot!(left_margin = 0px, right_margin = 0px, top_margin = 0px, bottom_margin = 40px)
    plot!(titlefontsize = 14, xtickfontsize = 12, ytickfontsize = 12, ztickfontsize = 12, xguidefontsize = 14, yguidefontsize = 14)
    plot!(title = "\n" * L"Pr(Goals_{A}=" * string(goalsA) * L", Goals_{B}=" * string(goalsB) * L"), \rho_{A,B}=" * strcorrAB)
    plot!(xlabel = "\n" * L"$\mu_{A}$")
    plot!(ylabel = "\n" * L"$\mu_{B}$")
    for points in pointslist
        plot!(points.xs, points.ys, zeros(length(points.xs)), color = :lightgrey, linewidth = 1, linestyle = :solid)
    end
    for points in pointslist
        plot!(points.xs, points.ys, points.zs, color = :blue, linewidth = linewidth, linestyle = :solid)
    end

    return plt
end

# ==============================================================================
# Main program
# ==============================================================================

p1 = subplot(0, 0, 0.0, L"0", 1)
p2 = subplot(0, 1, 0.0, L"0", 1)
p3 = subplot(1, 1, 0.0, L"0", 1)
p4 = subplot(0, 0, 0.5, L"\sqrt{\frac{1}{4}}", 1)
p5 = subplot(0, 1, 0.5, L"\sqrt{\frac{1}{4}}", 1)
p6 = subplot(1, 1, 0.5, L"\sqrt{\frac{1}{4}}", 1)
p7 = subplot(0, 0, sqrt(0.5), L"\sqrt{\frac{1}{2}}", 1)
p8 = subplot(0, 1, sqrt(0.5), L"\sqrt{\frac{1}{2}}", 1)
p9 = subplot(1, 1, sqrt(0.5), L"\sqrt{\frac{1}{2}}", 1)
p10 = subplot(0, 0, sqrt(7/8), L"\sqrt{\frac{7}{8}}", 1)
p11 = subplot(0, 1, sqrt(7/8), L"\sqrt{\frac{7}{8}}", 1)
p12 = subplot(1, 1, sqrt(7/8), L"\sqrt{\frac{7}{8}}", 1)
p13 = subplot(0, 0, 1.0, L"1", 2)
p14 = subplot(0, 1, 1.0, L"1", 2)
p15 = subplot(1, 1, 1.0, L"1", 2)

plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, layout = (5, 3))
plot!(size = (1200, 1800))  # 4.0in x 6.0in @ 300 dpi
display(plt)
savefig("Plots.NotPaper\\ProbscoreFromMeans.png")

# ==============================================================================
end
