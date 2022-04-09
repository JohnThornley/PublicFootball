module PyPlotPredictionDistanceVsCorrelation
# ==============================================================================
# Plot the sqrt of the mean across all matches of
# the squared distance between the observed match goals and the predicted mean match goals
# for a range of goal scoring correlations between the two teams.
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/stable/
import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot2d!(plt)
using LaTeXStrings  # needed for L notation
using Statistics
using Printf

using PlotUtil

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", "football2015")

MAX_Y = 2.0

# ==============================================================================
# Main program
# ==============================================================================

correlation = readcorrelations(db_conn)

distance = Vector{Float64}(undef, length(correlation))
mindistance = Inf
mindistancecorr = NaN

c = 1
done = false
while !done
    corr = correlation[c]
    query = """
    SELECT
        FulltimeHomeScore,
        FulltimeAwayScore,
        HomeGoalsMean,
        AwayGoalsMean
    FROM
        OddsImpliedProbs
    LEFT JOIN
        ProbsImpliedMeans
    ON ProbsImpliedMeans.AwaywinProb3DP = OddsImpliedProbs.AwaywinProb3DP AND ProbsImpliedMeans.HomewinProb3DP = OddsImpliedProbs.HomewinProb3DP
    WHERE
        Correlation = $(corr)
    """
    df = DBInterface.execute(db_conn, query) |> DataFrame
    nummatches = nrow(df)
    observedhomegoals = Int64.(df.FulltimeHomeScore)
    observedawaygoals = Int64.(df.FulltimeAwayScore)
    predictedmeanhomegoals = Float64.(df.HomeGoalsMean)
    predictedmeanawaygoals = Float64.(df.AwayGoalsMean)

    squareddistance = (observedhomegoals .- predictedmeanhomegoals).^2 + (observedawaygoals .- predictedmeanawaygoals).^2
    distance[c] = sqrt(mean(squareddistance))
    println("corr = $(corr), distance = $(distance[c])")
    if distance[c] < mindistance
        global mindistance = distance[c]
        global mindistancecorr = corr
    end


    if c == length(correlation) || MAX_Y <= distance[c]
        global done = true
    else
        global c += 1
    end
end
println("distance[$(mindistancecorr)] = $(mindistance)")
mindistancecorr2dp = round(mindistancecorr, RoundNearestTiesUp, digits=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 3))

ax.plot(correlation[1:c], distance[1:c], color="blue", linestyle="solid", zorder=1)
ax.plot([mindistancecorr2dp, mindistancecorr2dp], [0.0, MAX_Y], color="grey", linestyle="dotted", label=L"$\rho_{A,B}$=" * @sprintf("%4.2f", mindistancecorr2dp), zorder=2)
ax.set_xlabel(L"$\rho_{A, B}$")
ax.set_ylabel("Root mean squared predicted" * L"$-$" * "observed goals")
ax.legend(loc="upper right", edgecolor="black", fancybox=false)
ax.set_xlim(0, 1)
ax.set_yticks([1.6:0.1:MAX_Y;])
ax.set_ylim(1.55, MAX_Y)
ax.grid(true)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\PredictionDistanceVsCorrelation.png")

# ==============================================================================
end
