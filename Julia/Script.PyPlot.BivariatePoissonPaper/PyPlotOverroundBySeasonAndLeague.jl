module PyPlotOverroundBySeasonAndLeague
# ==============================================================================
# Plot over-round bar-charts against season and year categories
# for all observed matches.
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/v1.0/
import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot2d!(plt)
using Statistics
using HypothesisTests  # See: https://juliastats.org/HypothesisTests.jl/stable/

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", nothing)

CONF_LEVEL = 0.99

# ==============================================================================
# Main program
# ==============================================================================

query = """
    SELECT
        SeasonYear,
        LeagueName,
        Overround
    FROM
        OddsImpliedProbs
    """
df = DBInterface.execute(db_conn, query) |> DataFrame
df.SeasonYear = Int64.(df.SeasonYear)
df.LeagueName = string.(df.LeagueName)
df.Overround = Float64.(df.Overround)

meanoverround = round(mean(df.Overround),  RoundNearestTiesUp, digits=3)

SeasonYears = sort(unique(df.SeasonYear))
SeasonNames = string.(SeasonYears) .* "-" .* lpad.(string.((SeasonYears .+ 1) .% 100), 2, "0")

dfbyseason = groupby(df[:, [:SeasonYear, :Overround]], :SeasonYear)
dfbyleague = groupby(df[:, [:LeagueName, :Overround]], :LeagueName)
meansbyseason = sort(combine(dfbyseason, :Overround => mean), :SeasonYear)
meansbyleague = sort(combine(dfbyleague, :Overround => mean), :LeagueName)

seasonyears = meansbyseason[:, :SeasonYear]
seasonnames = string.(seasonyears) .* "-" .* lpad.(string.((seasonyears .+ 1) .% 100), 2, "0")
seasonmeans = meansbyseason[:, :Overround_mean]
leaguemeans = meansbyleague[:, :Overround_mean]

maxmean = max(maximum(seasonmeans), maximum(leaguemeans))
maxy = round(maxmean, RoundUp, digits=2)

# Compute error bars using T-test (not currently plotted, but keeping for future reference)
seasonconfints = [confint(OneSampleTTest(seasondf.Overround), level = 0.99, tail = :both) for seasondf in dfbyseason]
lowerrbar = seasonmeans .- [ci[1] for ci in seasonconfints]
higherrbar = [ci[2] for ci in seasonconfints] .- seasonmeans

fig = plt.figure(figsize=(5, 2))
gridspec = fig.add_gridspec(ncols=2, nrows=1, wspace=0, width_ratios=[2, 1])
(ax1, ax2) = gridspec.subplots(sharey=true)  # See: https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html

ax1.bar([1:length(seasonmeans);], seasonmeans, tick_label=SeasonNames, color="dodgerblue", edgecolor="black", zorder=1)
ax1.plot([0, length(seasonmeans) + 1], [meanoverround, meanoverround], color="red", linestyle="dashed", linewidth=1, label="Mean = $(meanoverround)", zorder=2)
ax1.set_ylabel("Mean Over-round")
ax1.tick_params(bottom=false)
ax1.set_xlim(0, length(seasonmeans) + 1)
ax1.set_ylim(1.0, maxy)
ax1.grid(true, axis="y")
for label in ax1.xaxis.get_ticklabels() label.set_y(+0.04); label.set_rotation(90) end
ax1.spines["top"].set_visible(false)
ax1.spines["right"].set_visible(false)

ax2.bar([1:length(leaguemeans);], leaguemeans, tick_label=["EPL", "ELC", "EL1", "EL2"], color="dodgerblue", edgecolor="black", zorder=1)
ax2.plot([0.35, length(leaguemeans) + 0.65], [meanoverround, meanoverround], color="red", linestyle="dashed", linewidth=1, label="Mean = $(meanoverround)", zorder=2)
ax2.legend(loc="upper right", edgecolor="black", fancybox=false)
ax2.tick_params(left=false, bottom=false)
ax2.set_xlim(0.35, length(leaguemeans) + 0.65)
ax2.set_ylim(1.0, maxy)
ax2.grid(true, axis="y")
for label in ax2.xaxis.get_ticklabels() label.set_y(+0.03) end
ax2.spines["top"].set_visible(false)
ax2.spines["right"].set_visible(false)

fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\OverroundBySeasonAndLeague.png", )

# ==============================================================================
end
