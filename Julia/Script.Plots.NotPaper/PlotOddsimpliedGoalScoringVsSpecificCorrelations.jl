module PlotOddsimpliedGoalScoringVsSpecificCorrelations
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
using Printf
using Plots
using Plots.PlotMeasures  # needed for px and mm notation in plots
using LaTeXStrings  # needed for L notation
using Statistics
using StatsPlots

using PlotUtil

pyplot()

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", nothing)

MAX_GOALS = 3  # Goals above 3 are grouped as >3

# ==============================================================================
# Main program
# ==============================================================================

homegoals, awaygoals = readmatchgoals(db_conn)
@assert length(homegoals) == length(awaygoals)
observedscore = observedscorefreq(homegoals, awaygoals, MAX_GOALS)

corrs = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
plts = Vector{Plots.Plot}(undef, length(corrs))
for i = 1:length(corrs)
    corr = corrs[i]
    homemean, awaymean = readmatchimpliedmeans(db_conn, corr)
    @assert length(homemean) == length(awaymean) == length(homegoals) == length(awaygoals)
    predictedscore = predictedscorefreq(homemean, awaymean, corr, MAX_GOALS)
    diffscore = predictedscore .- observedscore
    plts[i] = plotdiffscoreheatmap(diffscore, corr, title=L"$\rho_{A,B}$=" * @sprintf("%4.2f", corr), ylabel=((i - 1)%3 == 0), clim=(-2.5, 2.5), colors=[:red, :white, :green], colorbar=false)
end

plt = plot(plts[1], plts[2], plts[3], plts[4], plts[5], plts[6], layout = (2, 3))
plot!(size = (650, 450))  # 6.5in x 4.5in @ 100 dpi
display(plt)
savefig("Plots.NotPaper\\OddsimpliedGoalScoringVsSpecificCorrelations.png")

# ==============================================================================
end
