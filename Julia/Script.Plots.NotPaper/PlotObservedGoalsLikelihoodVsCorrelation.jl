module PlotObservedGoalsLikelihoodVsCorrelation
# ==============================================================================
# Plot the geometric mean across all matches of
# the likehood given by the model of the observed match goals
# for a range of goal scoring correlations between the two teams.
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/v1.0/
using Plots
using Plots.PlotMeasures  # needed for mm notation
using LaTeXStrings  # needed for L notation
using Statistics

using BivariatePoisson
using PlotUtil

pyplot()

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", nothing)

# ==============================================================================
# Main program
# ==============================================================================

correlation = readcorrelations(db_conn)

goalslikelihood = Vector{Float64}(undef, length(correlation))
maxgoalslikelihood = -1.0
maxgoalslikelihoodcorr = NaN

for (c, corr) in enumerate(correlation)
    query = """
    SELECT
        HomeGoalsMean,
        AwayGoalsMean,
        FulltimeHomeScore,
        FulltimeAwayScore
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
    meanhomegoals = Float64.(df.HomeGoalsMean)
    meanawaygoals = Float64.(df.AwayGoalsMean)
    observedhomegoals = Int64.(df.FulltimeHomeScore)
    observedawaygoals = Int64.(df.FulltimeAwayScore)
    for i in 1:length(meanhomegoals)
        if !meansvalid(meanhomegoals[i], meanawaygoals[i], corr)  # Rounding errors caused by truncation to 3 DP
            meanhomegoals[i] = if (meanhomegoals[i] < meanawaygoals[i]) minsmallermean(meanawaygoals[i], corr) else maxlargermean(meanawaygoals[i], corr) end
        end
    end

    probgoals = probscore_from_means.(observedhomegoals, observedawaygoals, meanhomegoals, meanawaygoals, corr)
    goalslikelihood[c] = exp(sum(log.(probgoals))/length(probgoals))
    if goalslikelihood[c] > maxgoalslikelihood
        global maxgoalslikelihood = goalslikelihood[c]
        global maxgoalslikelihoodcorr = corr
    end

    println("goalslikelihood[$(corr)] = $(goalslikelihood[c])")
end
println("goalslikelihood[$(maxgoalslikelihoodcorr)] = $(maxgoalslikelihood)")

plt = plot(xlim = (0, 1), xticks = [0.0:0.1:1.0;], ylim = (0, 0.06), yticks = [0.01:0.01:0.06;], legend = false)
plot!(left_margin = 0px, right_margin = 0px, top_margin = 5px, bottom_margin = 0px)
plot!(titlefontsize=8, xtickfontsize=7, ytickfontsize=7, xguidefontsize=8, yguidefontsize=8)
plot!(xlim=(0.0, 1.0), ylim=(0.0, 0.061))
plot!(xguide = L"$\rho_{A, B}$")
plot!(yguide = "Likelihood of observed match results")
#plot!(yguide = L"$^{n}\sqrt{\prod_{m \in all matches} Pr(Goals^{m}_{A}, Goals^{m}_{B})}$", titlefont="Serif")
plot!([maxgoalslikelihoodcorr, maxgoalslikelihoodcorr], [0.0, 1.0], color = :grey, linestyle = :dot)
plot!(correlation, goalslikelihood, color = :blue)

plot!(size = (300, 275))  # 2.75in x 3.0in @ 100 dpi
display(plt)
savefig("Plots.NotPaper\\ObservedGoalsLikelihoodVsCorrelation.png")

# ==============================================================================
end
