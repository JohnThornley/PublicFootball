module PlotProbresultsFromLambdas
# ==============================================================================
# Plot a grid of [Pr(winA), Pr(draw), Pr(winB)] vs (lambdaA, lambdaB) plots
# for different lambdaX
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using Plots
using Plots.PlotMeasures  # needed for px and mm notation in plots
using LaTeXStrings  # needed for L notation

using BivariatePoisson

pyplot()

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

MAX_LAMBDA = 4
LAMBDA_DIVISOR = 5
@assert 1 <= MAX_LAMBDA
@assert 1 <= LAMBDA_DIVISOR

CAMERA_ELEV = 30
CAMERA_AZIM = 125

# ==============================================================================
# Derived fixed values for this plot
# ==============================================================================

LAMBDA_RANGE = [0:1:(MAX_LAMBDA*LAMBDA_DIVISOR);]/LAMBDA_DIVISOR

# ==============================================================================
# Plotting
# ==============================================================================

function subplot(prob::Array{Float64, 2}, strresult::LaTeXString, lambdaX::Float64)::Plots.Plot
    @assert size(prob)[1] == length(LAMBDA_RANGE) && size(prob)[2] == length(LAMBDA_RANGE)
    @assert 0.0 <= minimum(prob) && maximum(prob) <= 1.0
    @assert 0 < length(strresult)
    @assert 0 <= lambdaX && !isinf(lambdaX)
    prob_result(lambdaA, lambdaB) = prob[LAMBDA_RANGE .== lambdaA, LAMBDA_RANGE .== lambdaB][1, 1]
    p = plot(LAMBDA_RANGE, LAMBDA_RANGE, prob_result, zlims = (0.0, 1.0), seriestype = :wireframe, color = :blue, linewidth = 1, linestyle = :solid)
    plot!(camera = (CAMERA_AZIM, CAMERA_ELEV))
    plot!(left_margin = 0px, right_margin = 0px, top_margin = 0px, bottom_margin = 40px)
    plot!(titlefontsize = 14, xtickfontsize = 12, ytickfontsize = 12, ztickfontsize = 12, xguidefontsize = 14, yguidefontsize = 14)
    plot!(title = "\n" * L"Pr(" * strresult * L"), \lambda_{X}=" * string(lambdaX))
    plot!(xlabel = "\n" * L"$\lambda_{A}$")
    plot!(ylabel = "\n" * L"$\lambda_{B}$")
    return p
end

function subplots(lambdaX::Float64)::Tuple{Plots.Plot, Plots.Plot, Plots.Plot}
    @assert 0 <= lambdaX && !isinf(lambdaX)
    probwinA::Array{Float64, 2} = fill(NaN, length(LAMBDA_RANGE), length(LAMBDA_RANGE))
    probwinB::Array{Float64, 2} = fill(NaN, length(LAMBDA_RANGE), length(LAMBDA_RANGE))
    for (i, lambdaA) in enumerate(LAMBDA_RANGE)
        for (j, lambdaB) in enumerate(LAMBDA_RANGE)
            probwinB[i, j], probwinA[i, j] = probresults_from_lambdas(lambdaA, lambdaB, lambdaX)
    end end
    probdraw::Array{Float64, 2} = 1.0 .- probwinA .- probwinB
    plotwinA = subplot(probwinA, L"win_A", lambdaX)
    plotdraw = subplot(probdraw, L"draw", lambdaX)
    plotwinB = subplot(probwinB, L"win_B", lambdaX)
    return plotwinA, plotdraw, plotwinB
end

# ==============================================================================
# Main program
# ==============================================================================

p1, p2, p3 = subplots(0.0)
p4, p5, p6 = subplots(1.0)

plt = plot(p1, p2, p3, p4, p5, p6, layout = (2, 3))
plot!(size = (1350, 900))  # 4.5in x 3.0in @ 300 dpi
display(plt)
savefig("Plots.NotPaper\\ProbresultsFromLambdas.png")

# ==============================================================================
end
