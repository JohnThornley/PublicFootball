module PyPlotOddsimpliedGoalScoringVsCorrelation
# ==============================================================================
# Plot 2-by-3 grid of
# heatmaps of predicted vs observed goals and
# density functions of odds-implied goal-scoring rates for
# correlation coefficients of 0.0, 0.1, and 0.3
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/stable/
using OffsetArrays  # https://github.com/JuliaArrays/OffsetArrays.jl
import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot2d!(plt)
using LaTeXStrings  # needed for L notation
using Printf
using Statistics
using KernelDensity

using PlotUtil

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", nothing)

CORRELATIONS = [0.0, 0.15, 0.3]
MAX_GOALS = 3  # Goals above 3 are grouped as >3

# ==============================================================================
# Plot functions
# ==============================================================================

function plotgoalscoringdensity!(ax, observedhomemean::Float64, observedawaymean::Float64, predictedhomemean::Vector{Float64}, predictedawaymean::Vector{Float64}, corr::Float64, ylabel::Bool, legend::Bool)::Nothing
    homedensity = kde(predictedhomemean)
    awaydensity = kde(predictedawaymean)
    ax.plot(homedensity.x, homedensity.density, color="blue", label=if legend "Home" else nothing end)
    ax.plot(awaydensity.x, awaydensity.density, color="red", label=if legend "Away" else nothing end)
    avgpredictedhomemean = mean(predictedhomemean)
    ax.plot([avgpredictedhomemean, avgpredictedhomemean], [0, 10], color="blue", linewidth=1, linestyle="dashed")
    avgpredictedawaymean = mean(predictedawaymean)
    ax.plot([avgpredictedawaymean, avgpredictedawaymean], [0, 10], color="red", linewidth=1, linestyle="dashed")
    ax.set_title(L"$\rho_{A,B}$ = "*@sprintf("%4.2f", corr))
    ax.set_xlabel("Odds-implied goal scoring rate")
    if ylabel ax.set_ylabel("Probability Density"); ax.yaxis.set_label_coords(-0.05, 0.5) end
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.2)
    ax.set_yticks([])
    ax.grid(true, axis="x")
    if legend ax.legend(loc="upper left", edgecolor="black", fancybox=false) end
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    # plt = plot(title=L"$\rho_{A,B}$=" * @sprintf("%4.2f", corr), xlabel="Odds-implied goal scoring rate", ylabel=if ylabel "Probability" else nothing end)
    # plot!(left_margin=0px, right_margin=0px, top_margin=0px, bottom_margin=20px, legend=:topleft)
    # plot!(titlefontsize=9, xtickfontsize=7, ytickfontsize=7, xguidefontsize=8, yguidefontsize=8, legendfontsize=5.5)
    # density!(predictedhomemean, xlim=(0, 2.5), ylim=(0, 2.2), yticks=1:1:2, ytickfontcolor=:white, color=:blue, label=if legend "Home" else nothing end)
    # #plot!([observedhomemean, observedhomemean], [0, 10], color=:blue, linewidth=1, linestyle=:dot, label=nothing)
    # plot!([avgpredictedhomemean, avgpredictedhomemean], [0, 10], color=:blue, linewidth=1, linestyle=:dash, label=nothing)
    # density!(predictedawaymean, color=:red, label=if legend "Away" else nothing end)
    # #plot!([observedawaymean, observedawaymean], [0, 10], color=:red, linewidth=1, linestyle=:dot, label=nothing)
    # plot!([avgpredictedawaymean, avgpredictedawaymean], [0, 10], color=:red, linewidth=1, linestyle=:dash, label=nothing)
    return
end

# ==============================================================================
# Main program
# ==============================================================================

homegoals, awaygoals = readmatchgoals(db_conn)
@assert length(homegoals) == length(awaygoals)
observedscore = observedscorefreq(homegoals, awaygoals, MAX_GOALS)

rdwhgn = plt.LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])

fig, axs = plt.subplots(ncols=length(CORRELATIONS), nrows=2, figsize=(6, 4))

for i = 1:length(CORRELATIONS)
    corr = CORRELATIONS[i]
    homemean, awaymean = readmatchimpliedmeans(db_conn, corr)
    @assert length(homemean) == length(awaymean) == length(homegoals) == length(awaygoals)
    plotgoalscoringdensity!(axs[2*(i - 1) + 1], mean(homegoals), mean(awaygoals), homemean, awaymean, corr, i == 1, i==3)
    predictedscore = predictedscorefreq(homemean, awaymean, corr, MAX_GOALS)
    diffscore = predictedscore - observedscore
    plotscoreheatmap!(axs[2*(i - 1) + 2], diffscore, corr, title=" ", ylabel=(i == 1), clim=(-3.0, 3.0), cmap=rdwhgn)
end

fig.tight_layout(pad=0.75, w_pad=2.0)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\OddsimpliedGoalScoringVsCorrelation.png")

# ==============================================================================
end
