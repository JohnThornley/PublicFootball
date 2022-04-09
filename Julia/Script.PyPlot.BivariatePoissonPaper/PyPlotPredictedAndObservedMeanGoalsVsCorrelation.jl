module PyPlotPredictedAndObservedMeanGoalsVsCorrelation
# ==============================================================================
# Plot the expected home and away goals from the model for the sample matches
# over a range of goal scoring correlations between the two teams.
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

db_conn = ODBC.Connection("football", "football", nothing)

MAX_Y = 4

# ==============================================================================
# Main program
# ==============================================================================

correlation = readcorrelations(db_conn)

query = """
    SELECT
      AVG(Convert(Float, FulltimeHomeScore)) AS AvgFulltimeHomeScore,
      AVG(Convert(Float, FulltimeAwayScore)) AS AvgFulltimeAwayScore
    FROM OddsImpliedProbs
    """
df = DBInterface.execute(db_conn, query) |> DataFrame
samplemeangoalshome = df.AvgFulltimeHomeScore[1]
samplemeangoalsaway = df.AvgFulltimeAwayScore[1]

expectedgoalshome = Vector{Float64}(undef, length(correlation))
expectedgoalsaway = Vector{Float64}(undef, length(correlation))
mindiff = Inf
mindiffcorr = NaN

c = 1
done = false
while !done
    corr = correlation[c]
    query = """
    SELECT
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
    expectedgoalshome[c] = mean(Float64.(df.HomeGoalsMean))
    expectedgoalsaway[c] = mean(Float64.(df.AwayGoalsMean))

    diff = sqrt((expectedgoalshome[c] - samplemeangoalshome)^2 + (expectedgoalsaway[c] - samplemeangoalsaway)^2)
    if diff < mindiff
        global mindiff = diff
        global mindiffcorr = corr
    end

    println("expectedgoalshome[$(corr)] = $(expectedgoalshome[c]), expectedgoalsaway[$(corr)] = $(expectedgoalsaway[c]), diff = $(diff)")
    if c == length(correlation) || MAX_Y <= expectedgoalsaway[c]
        global done = true
    else
        global c += 1
    end
end
println("diff[$(mindiffcorr)] = $(mindiff)")
mindiffcorr2dp = round(mindiffcorr, RoundNearestTiesUp, digits=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 2.75))

ax.plot(correlation[1:c], expectedgoalshome[1:c], color="blue", linestyle="solid", label="Predicted home", zorder=3)
ax.plot(correlation[1:c], expectedgoalsaway[1:c], color="red", linestyle="solid", label="Predicted away", zorder=3)
ax.plot([0.0, 1.0], [samplemeangoalshome, samplemeangoalshome], color="blue", linestyle="dashed", label="Observed home", zorder=2)
ax.plot([0.0, 1.0], [samplemeangoalsaway, samplemeangoalsaway], color="red", linestyle="dashed", label="Observed away", zorder=2)
ax.plot([mindiffcorr, mindiffcorr], [0.0, MAX_Y], color="grey", linestyle="dotted", label = L"$\rho_{A, B}$ = " * @sprintf("%4.2f", mindiffcorr2dp), zorder=1)
ax.set_xlabel(L"$\rho_{A, B}$")
ax.set_ylabel("Mean goals")
ax.legend(loc="upper left", edgecolor="black", fancybox=false)
ax.set_xlim(0, 1)
ax.set_yticks([0:MAX_Y;])
ax.set_ylim(0, MAX_Y)
ax.grid(true)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\PredictedAndObservedMeanGoalsVsCorrelation.png")

# ==============================================================================
end
