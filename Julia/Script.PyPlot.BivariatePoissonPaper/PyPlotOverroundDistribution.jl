module PyPlotOverroundDistribution
# ==============================================================================
# Plot the distribution (density function) of
# over-rounds for best bookmaker match result odds.
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/v1.0/
import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot2d!(plt)
using Statistics

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", "football2015")

BAR_WIDTH = 0.005

# ==============================================================================
# Main program
# ==============================================================================

query = """
    SELECT
        Overround
    FROM OddsImpliedProbs
    """
df = DBInterface.execute(db_conn, query) |> DataFrame
maxoverround = maximum(df.Overround)
meanoverround = round(mean(df.Overround), RoundNearestTiesUp, digits=3)
topofrange = round(maxoverround , RoundNearestTiesUp, digits=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 2))

ax.hist(df.Overround, bins=[1.0:BAR_WIDTH:topofrange;], color="dodgerblue", edgecolor="black", zorder=1)
ymax = ax.get_ylim()[2]
ax.plot([meanoverround, meanoverround], [0, ymax], color="red", linestyle="dashed", label="Mean = $(meanoverround)", zorder=2)
ax.set_xlabel("Over-round")
ax.set_ylabel("Frequency")
ax.legend(loc="upper right", edgecolor="black", fancybox=false)
ax.set_xticks([1.0:(4*BAR_WIDTH):topofrange;])
ax.set_xlim(1.0 - BAR_WIDTH/2, topofrange + BAR_WIDTH/2)
ax.set_yticks([])
ax.set_ylim(0, ymax)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\OverroundDistribution.png")

# ==============================================================================
end
