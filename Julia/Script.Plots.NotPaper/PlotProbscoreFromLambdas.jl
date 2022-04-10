module PlotProbscoreFromLambdas
# ==============================================================================
# Plot a grid of Pr(goalsA, goalsB) vs (lambdaA, lambdaB) plots
# each for different goalsA, goalsB, and lambdaX
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using Plots
using Plots.PlotMeasures  # needed for mm notation
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

function subplot(goalsA::Int64, goalsB::Int64, lambdaX::Float64)::Plots.Plot
    @assert 0 <= goalsA
    @assert 0 <= goalsB
    @assert 0.0 <= lambdaX && !isinf(lambdaX)
    prob_goalsAgoalsB(lambdaA, lambdaB) = probscore_from_lambdas(goalsA, goalsB, lambdaA, lambdaB, lambdaX)
    plt = plot(LAMBDA_RANGE, LAMBDA_RANGE, prob_goalsAgoalsB, zlim = (0.0, 1.0), legend = false, seriestype = :wireframe, color = :blue, linewidth = 1, linestyle = :solid)
    plot!(camera = (CAMERA_AZIM, CAMERA_ELEV))
    plot!(left_margin = 0px, right_margin = 0px, top_margin = 0px, bottom_margin = 40px)
    plot!(titlefontsize = 14, xtickfontsize = 12, ytickfontsize = 12, ztickfontsize = 12, xguidefontsize = 14, yguidefontsize = 14)
    plot!(title = "\n" * L"Pr(Goals_{A}=" * string(goalsA) * L", Goals_{B}=" * string(goalsB) * L"), \lambda_{X}=" * string(lambdaX))
    plot!(xlabel = "\n" * L"$\lambda_{A}$")
    plot!(ylabel = "\n" * L"$\lambda_{B}$")
    return plt
end

# ==============================================================================
# Main program
# ==============================================================================

p1 = subplot(0, 0, 0.0)
p2 = subplot(0, 1, 0.0)
p3 = subplot(1, 1, 0.0)
p4 = subplot(0, 0, 0.5)
p5 = subplot(0, 1, 0.5)
p6 = subplot(1, 1, 0.5)
p7 = subplot(0, 0, 1.0)
p8 = subplot(0, 1, 1.0)
p9 = subplot(1, 1, 1.0)
p10 = subplot(0, 0, 2.0)
p11 = subplot(0, 1, 2.0)
p12 = subplot(1, 1, 2.0)

plt = plot(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, layout = (4, 3))
plot!(size = (1200, 1440))  # 4.5in x 4.8in @ 300 dpi
display(plt)
savefig("Plots.NotPaper\\ProbscoreFromLambdas.png")

# ==============================================================================
end
