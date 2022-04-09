module PyPlotOddsimpliedGoalScoringVsOneCorrelation
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

using PlotUtil

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", nothing)

CORR = 0.13
MAX_GOALS = 3  # Goals above 3 are grouped as >3

# ==============================================================================
# Main program
# ==============================================================================

homegoals, awaygoals = readmatchgoals(db_conn)
@assert length(homegoals) == length(awaygoals)
homemean, awaymean = readmatchimpliedmeans(db_conn, CORR)
@assert length(homemean) == length(awaymean) == length(homegoals) == length(awaygoals)

whbl = plt.LinearSegmentedColormap.from_list("WhBl", ["white", "dodgerblue"])
rdwhgn = plt.LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(5.75, 2.2))
corrtitle = L"$\rho_{A,B}$ = "*@sprintf("%4.2f", CORR)
# fig.suptitle(corrtitle)

predictedscore = predictedscorefreq(homemean, awaymean, CORR, MAX_GOALS)
plotscoreheatmap!(ax1, predictedscore, CORR, title="\n\nPredicted", ylabel=true, clim=(0.0, 20.0), cmap=whbl)

observedscore = observedscorefreq(homegoals, awaygoals, MAX_GOALS)
plotscoreheatmap!(ax2, observedscore, CORR, title=corrtitle*"\n\nObserved", ylabel=false, clim=(0.0, 20.0), cmap=whbl)

diffscore = predictedscore .- observedscore
plotscoreheatmap!(ax3, diffscore, CORR, title="\n\nPredicted"*L"$-$"*"Observed", ylabel=false, clim=(-3.0, 3.0), cmap=rdwhgn)

fig.tight_layout(pad=0.75, w_pad=3.5)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\OddsimpliedGoalScoringVsOneCorrelation.png")

# ==============================================================================
end
