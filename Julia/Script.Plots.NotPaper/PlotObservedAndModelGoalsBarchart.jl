module PlotObservedAndModelGoalsBarchart
# ==============================================================================
# Plot probability bar-charts for
# observed and modeled total goals and goal differences
# for all observed matches.
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/v1.0/
using Plots
using StatsPlots
using Plots.PlotMeasures  # needed for mm notation
using LaTeXStrings  # needed for L notation
using CategoricalArrays

using BivariatePoisson

pyplot()

# ==============================================================================
# Fixed values for this plot
# ==============================================================================

MIN_GOALSUM = 0
MAX_GOALSUM = 10
GOALSUMS = [MIN_GOALSUM:MAX_GOALSUM;]

MIN_GOALDIFF = -5
MAX_GOALDIFF = +5
GOALDIFFS = [MIN_GOALDIFF:MAX_GOALDIFF;]

OBSERVED = "Observed"
MODEL00 = L"Bivariate Poisson, $\rho_{home, away}=0.0$"
MODEL12 = L"Bivariate Poisson, $\rho_{home, away}=0.12$"

db_conn = ODBC.Connection("football", "football", nothing)

# ==============================================================================
# Main program
# ==============================================================================

query = """
    SELECT
        COUNT(1) AS Count,
        AVG(CONVERT(FLOAT, FulltimeHomeScore)) AS AvgHomeGoals,
        AVG(CONVERT(FLOAT, FulltimeAwayScore)) AS AvgAwayGoals
    FROM
        OddsImpliedProbs
    """
df = DBInterface.execute(db_conn, query) |> DataFrame
numobservations = df.Count[1]
observedavghomegoals = df.AvgHomeGoals[1]
observedavgawaygoals = df.AvgAwayGoals[1]

println("numobservations = $(numobservations), observedavghomegoals = $(observedavghomegoals), observedavgawaygoals = $(observedavgawaygoals)")

query = """
    SELECT
        CONVERT(INT, FulltimeHomeScore) + CONVERT(INT, FulltimeAwayScore) AS GoalSum,
        COUNT(1) AS Count
    FROM
        OddsImpliedProbs
    GROUP BY
        CONVERT(INT, FulltimeHomeScore) + CONVERT(INT, FulltimeAwayScore)
    ORDER BY
        CONVERT(INT, FulltimeHomeScore) + CONVERT(INT, FulltimeAwayScore)
    """
df = DBInterface.execute(db_conn, query) |> DataFrame

@assert sum(df.Count) == numobservations
observedgoalsumprob = zeros(length(GOALSUMS))
for (gs, count) in zip(df.GoalSum, df.Count)
    if gs in GOALSUMS
        i = (gs - MIN_GOALSUM) + 1
        observedgoalsumprob[i] = count/numobservations
    end
end

query = """
    SELECT
        CONVERT(INT, FulltimeHomeScore) - CONVERT(INT, FulltimeAwayScore) AS GoalDiff,
        COUNT(1) AS Count
    FROM
        OddsImpliedProbs
    GROUP BY
        CONVERT(INT, FulltimeHomeScore) - CONVERT(INT, FulltimeAwayScore)
    ORDER BY
        CONVERT(INT, FulltimeHomeScore) - CONVERT(INT, FulltimeAwayScore)
    """
df = DBInterface.execute(db_conn, query) |> DataFrame

@assert sum(df.Count) == numobservations
observedgoaldiffprob = zeros(length(GOALDIFFS))
for (gd, count) in zip(df.GoalDiff, df.Count)
    if gd in GOALDIFFS
        i = (gd - MIN_GOALDIFF) + 1
        observedgoaldiffprob[i] = count/numobservations
    end
end

function modelprobs(corr::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    modelgoalsumprob = zeros(length(GOALSUMS))
    modelgoaldiffprob = zeros(length(GOALSUMS))

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
    nummatches = nrow(df)
    modelmeangoalshome = Float64.(df.HomeGoalsMean)
    modelmeangoalsaway = Float64.(df.AwayGoalsMean)
    for i in 1:length(modelmeangoalshome)
        if !meansvalid(modelmeangoalshome[i], modelmeangoalsaway[i], corr)  # Rounding errors caused by truncation to 3 DP
            modelmeangoalshome[i] = if (modelmeangoalshome[i] < modelmeangoalsaway[i]) minsmallermean(modelmeangoalsaway[i], corr) else maxlargermean(modelmeangoalsaway[i], corr) end
        end
    end

    for (meangoalshome, meangoalsaway) in zip(modelmeangoalshome, modelmeangoalsaway)
        totalprob = 0.0
        maxgoals = 0
        while totalprob < 0.999
            goals = vcat([0:maxgoals;], repeat([maxgoals], maxgoals))
            for (goalshome, goalsaway) in zip(goals, reverse(goals))
                probscore = probscore_from_means(goalshome, goalsaway, meangoalshome, meangoalsaway, corr)
                totalprob += probscore
                if MIN_GOALSUM <= goalshome + goalsaway < MAX_GOALSUM
                    modelgoalsumprob[goalshome + goalsaway - MIN_GOALSUM + 1] += probscore
                end
                if MIN_GOALDIFF <= goalshome - goalsaway <= MAX_GOALDIFF
                    modelgoaldiffprob[goalshome - goalsaway - MIN_GOALDIFF + 1] += probscore
                end
            end
            maxgoals += 1
        end
    end

    return (modelgoalsumprob, modelgoaldiffprob)./nummatches
end

function Base.unique(ctg::CategoricalArray)
# See: https://discourse.julialang.org/t/statplots-groupedbar-order-x-axis/13912/18
    l = levels(ctg)
    newctg = CategoricalArray(l)
    levels!(newctg, l)
end

names = CategoricalArray(repeat([OBSERVED, MODEL00, MODEL12], inner = length(GOALSUMS)))
levels!(names, [OBSERVED, MODEL00, MODEL12])

model00goalsumprob, model00goaldiffprob = modelprobs(0.00)
model12goalsumprob, model12goaldiffprob = modelprobs(0.12)

goalsumprobs = vcat(observedgoalsumprob, model00goalsumprob, model12goalsumprob)
goalsums = vcat(GOALSUMS, GOALSUMS, GOALSUMS)
goalsumplot = groupedbar(goalsums, goalsumprobs, group = names, xticks = GOALSUMS, ylim = (0.0, 0.33), xguide = L"$Goal_{home} + Goals_{away}$", yguide = L"$Prob$")
plot!(left_margin = 10px, right_margin = 20px, top_margin = 10px, bottom_margin = 10px)
plot!(titlefontsize = 14, xtickfontsize = 12, ytickfontsize = 12, xguidefontsize = 16, yguidefontsize = 14, legendfontsize=12)

goaldiffprobs = vcat(observedgoaldiffprob, model00goaldiffprob, model12goaldiffprob)
goaldiffs = vcat(GOALDIFFS, GOALDIFFS, GOALDIFFS)
goaldiffplot = groupedbar(goaldiffs, goaldiffprobs, group = names, xticks = GOALDIFFS, ylim = (0.0, 0.33), xguide = L"$Goals_{home} - Goals_{away}$", yguide = L"$Prob$")
plot!(left_margin = 20px, right_margin = 10px, top_margin = 10px, bottom_margin = 10px)
plot!(titlefontsize = 14, xtickfontsize = 12, ytickfontsize = 12, xguidefontsize = 16, yguidefontsize = 14, legendfontsize=12)

plt = plot(goalsumplot, goaldiffplot, layout = (1, 2))
plot!(size = (1350, 750))  # 4.5in x 2.5in @ 300 dpi

plt = plot(goalsumplot, goaldiffplot, layout = (1, 2))
plot!(size = (1200, 660))  # 4.0in x 2.2in @ 300 dpi
display(plt)
savefig("Plots.NotPaper\\ObservedAndModelGoalsBarchart.png")

# ==============================================================================
end
