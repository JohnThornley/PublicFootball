module PlotLambdasFromProbresults
# ==============================================================================
# Plot lambdaA vs [Pr(winA), Pr(winB)] and lambdaB vs [Pr(winA), Pr(winB)] plots
# for a given lambdaX
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using Plots
using Plots.PlotMeasures  # needed for mm notation
using LaTeXStrings  # needed for L notation

using BivariatePoisson
using Util

pyplot()

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

PROB_DIVISOR = 20
@assert PROB_DIVISOR % 2 == 0

MAX_LAMBDA = 4
@assert 1 <= MAX_LAMBDA

CAMERA_ELEV = 30
CAMERA_AZIM = 300

LAMBDA_TOLERANCE = 1e-04

# ==============================================================================
# Derived fixed values for this plot
# ==============================================================================

PROB_RANGE = [0:1:PROB_DIVISOR;]/PROB_DIVISOR
LAMBDA_RANGE = [0:1:MAX_LAMBDA;]
LAMBDA_LEVELS = Float64.(LAMBDA_RANGE[2:length(LAMBDA_RANGE)])

# ==============================================================================
# Plotting
# ==============================================================================

function subplot(pointslist::Vector{Points3D}, strresult::LaTeXString, lambdaX::Float64)::Plots.Plot
    zs = Iterators.flatten(points.zs for points in pointslist)
    @assert 0.0 <= minimum(zs)
    @assert maximum(zs) <= MAX_LAMBDA
    @assert 0 < length(strresult)
    @assert 0 <= lambdaX
    plt = plot(xlim = (0.0, 1.0), ylim = (0.0, 1.0), zlim = (0, MAX_LAMBDA), legend = false)
    plot!(camera = (CAMERA_AZIM, CAMERA_ELEV))
    plot!(left_margin = 20px, right_margin = 10px, top_margin = 0px, bottom_margin = 0px)
    plot!(titlefontsize = 16, xtickfontsize = 14, ytickfontsize = 14, ztickfontsize = 14, xguidefontsize = 16, yguidefontsize = 16)
    plot!(title = strresult * " " * L"$(\lambda_{X}=$" * string(lambdaX) * L"$)$")
    plot!(xlabel = "\n\n" * L"$Pr(win_{A})$")
    plot!(ylabel = "\n\n" * L"$Pr(win_{B})$")
    for points in pointslist
        plot!(points.xs, points.ys, zeros(length(points.xs)), color = :lightgrey, linewidth = 1, linestyle = :solid)
    end
    for points in pointslist
        plot!(points.xs, points.ys, points.zs, color = :blue, linewidth = 1, linestyle = :solid)
    end
    return plt
end

function subplots(lambdaX::Float64)::Tuple{Plots.Plot, Plots.Plot}
    @assert 0 <= lambdaX && !isinf(lambdaX)

    lambdaA::Array{Float64, 2} = fill(NaN, length(PROB_RANGE), length(PROB_RANGE))
    lambdaB::Array{Float64, 2} = fill(NaN, length(PROB_RANGE), length(PROB_RANGE))
    for (i, probwinA) in enumerate(PROB_RANGE)
        for (j, probwinB) in enumerate(PROB_RANGE)
            if probwinA + probwinB < 1.0
                lambdaA[i, j], lambdaB[i, j] = lambdas_from_probresults(probwinB, probwinA, lambdaX, LAMBDA_TOLERANCE)
    end end end

    pointslistlambdaA::Vector{Points3D} = []
    pointslistlambdaB::Vector{Points3D} = []

    horizontalLevelslambdaA::Vector{Points3D} = [Points3D([], [], []) for level in 1:length(LAMBDA_LEVELS)]
    horizontalLevelslambdaB::Vector{Points3D} = [Points3D([], [], []) for level in 1:length(LAMBDA_LEVELS)]

    for (k, probwin) in enumerate(PROB_RANGE)
        println("k = $(k), probwin = $(probwin)")

        # Lines for fixed Pr(winA), varying Pr(winB) on lambdaA plot
        validj::Vector{Bool} = lambdaA[k, :] .< MAX_LAMBDA
        bs::Vector{Float64} = PROB_RANGE[validj]
        as::Vector{Float64} = fill(PROB_RANGE[k], length(bs))
        fixedwinAlinelambdaA = Points3D(as, bs, lambdaA[k, validj])
        if (k == 1) push!(fixedwinAlinelambdaA, 0.0, 1.0, 0.0) end # add corner

        # Lines for fixed Pr(winA), varying Pr(winB) on lambdaB plot
        validj = lambdaB[k, :] .< MAX_LAMBDA
        bs = PROB_RANGE[validj]
        as = fill(PROB_RANGE[k], length(bs))
        fixedwinAlinelambdaB = Points3D(as, bs, lambdaB[k, validj])

        # Lines for fixed Pr(winB), varying Pr(winA) on lambdaA plot
        validi::Vector{Bool} = lambdaA[:,k] .< MAX_LAMBDA
        as = PROB_RANGE[validi]
        bs = fill(PROB_RANGE[k], length(as))
        fixedwinBlinelambdaA = Points3D(as, bs, lambdaA[validi, k])

        # Lines for fixed Pr(winB), varying Pr(winA) on lambdaB plot
        validi = lambdaB[:,k] .< MAX_LAMBDA
        as = PROB_RANGE[validi]
        bs = fill(PROB_RANGE[k], length(as))
        fixedwinBlinelambdaB = Points3D(as, bs, lambdaB[validi, k])
        if (k == 1) push!(fixedwinBlinelambdaB, 1.0, 0.0, 0.0) end # add corner

        # Append to horizontal lines for each LAMBDA_LEVEL
        for (level, levellambda) in enumerate(LAMBDA_LEVELS)
            println("level = $(level), levellambda = $(levellambda)")
            println("proby1gty2_from_proby1lty2andlambda1(probwin = $(probwin), levellambda = $(levellambda), lambdaX, LAMBDA_TOLERANCE)")
            probwinAFromB = if (probwin == 1.0) 0.0 else proby1gty2_from_proby1lty2andlambda1(probwin, levellambda, lambdaX, LAMBDA_TOLERANCE) end
            push!(fixedwinBlinelambdaA, probwinAFromB, probwin, levellambda)
            push!(fixedwinAlinelambdaB, probwin, probwinAFromB, levellambda)
            push!(horizontalLevelslambdaA[level], probwinAFromB, probwin, levellambda)
            push!(horizontalLevelslambdaB[level], probwin, probwinAFromB, levellambda)
            println("proby1lty2_from_proby1gty2andlambda1(probwin = $(probwin), levellambda = $(levellambda), lambdaX, LAMBDA_TOLERANCE)")
            probwinBFromA = if (probwin == 1.0) NaN elseif (probwin == 0.0) 1.0 else proby1lty2_from_proby1gty2andlambda1(probwin, levellambda, lambdaX, LAMBDA_TOLERANCE) end
            if !isnan(probwinBFromA)
                push!(fixedwinAlinelambdaA, probwin, probwinBFromA, levellambda)
                push!(fixedwinBlinelambdaB, probwinBFromA, probwin, levellambda)
                push!(horizontalLevelslambdaA[level], probwin, probwinBFromA, levellambda)
                push!(horizontalLevelslambdaB[level], probwinBFromA, probwin, levellambda)
            end
        end

        # Sort and append grid lines
        push!(pointslistlambdaA, sorted(fixedwinAlinelambdaA, fixedwinAlinelambdaA.ys))
        push!(pointslistlambdaB, sorted(fixedwinAlinelambdaB, fixedwinAlinelambdaB.ys))
        push!(pointslistlambdaA, sorted(fixedwinBlinelambdaA, fixedwinBlinelambdaA.xs))
        push!(pointslistlambdaB, sorted(fixedwinBlinelambdaB, fixedwinBlinelambdaB.xs))
    end

    # Sort and append horizontal lines for each LAMBDA_RANGE
    for level in 1:length(LAMBDA_LEVELS)
        push!(pointslistlambdaA, sorted(horizontalLevelslambdaA[level], (1.0 .- horizontalLevelslambdaA[level].ys)./(1.0 .- horizontalLevelslambdaA[level].xs)))
        push!(pointslistlambdaB, sorted(horizontalLevelslambdaB[level], (1.0 .- horizontalLevelslambdaB[level].xs)./(1.0 .- horizontalLevelslambdaB[level].ys)))
    end

    plotlambdaA = subplot(pointslistlambdaA, L"$\lambda_{A}$", lambdaX)
    plotlambdaB = subplot(pointslistlambdaB, L"$\lambda_{B}$", lambdaX)
    return plotlambdaA, plotlambdaB
end

# ==============================================================================
# Main program
# ==============================================================================

lambda1plot, lambda2plot = subplots(0.0)
plt = plot(lambda1plot, lambda2plot, layout = (1, 2))
plot!(size = (1350, 750))  # 4.5in x 2.5in @ 300 dpi
display(plt)
savefig("Plots.NotPaper\\LambdasFromProbresults.png")

# ==============================================================================
end
