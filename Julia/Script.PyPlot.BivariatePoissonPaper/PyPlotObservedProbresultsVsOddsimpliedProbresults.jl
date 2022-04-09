module PyPlotObservedProbresultsVsOddsimpliedProbresults
# ==============================================================================
# Plot binned scatterplot of observed prob win vs raw odds-implied prob win.
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/stable/
import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot2d!(plt)
using LaTeXStrings  # needed for L notation
using GLM  # See: https://juliastats.org/GLM.jl/stable/
using Printf

# ============================================================================  ==
# Fixed values for this plot
# ==============================================================================

db_conn = ODBC.Connection("football", "football", nothing)

plt.rc("legend", fontsize=6)

# ==============================================================================
# Functions
# ==============================================================================

function binnify(count::Vector{Int64}, x::Vector{Float64}, y::Vector{Float64}, numbins::Int64)::Tuple{Vector{Float64}, Vector{Float64}}
    @assert 0 < length(count) == length(x) == length(y)
    @assert 0 < numbins
    global xbin = Vector{Float64}(undef, numbins)
    global ybin = Vector{Float64}(undef, numbins)

    totalcount = sum(count)
    cummcount = [sum(count[1:c]) for c in 1:length(count)]

    binstart = [(round(totalcount*b/numbins, RoundNearestTiesUp) + 1) for b in 0:numbins]
    binend = [binstart[b + 1] - 1 for b in 1:numbins]
    bincount = binend .- binstart[1:numbins] .+ 1
    @assert sum(bincount) == totalcount

    global startindex = 1
    for bin in 1:numbins
        while cummcount[startindex] < binstart[bin] startindex += 1 end
        endindex = startindex
        while cummcount[endindex] < binend[bin] endindex += 1 end
        @assert 1 <= startindex <= endindex <= length(count)

        fullrange = startindex:endindex
        xsum = sum(x[fullrange].*count[fullrange])
        ysum = sum(y[fullrange].*count[fullrange])
        xsum -= x[startindex]*(count[startindex] - (cummcount[startindex] - binstart[bin] + 1))
        ysum -= y[startindex]*(count[startindex] - (cummcount[startindex] - binstart[bin] + 1))
        xsum -= x[endindex]*(cummcount[endindex] - binend[bin])
        ysum -= y[endindex]*(cummcount[endindex] - binend[bin])
        @assert sum(count[fullrange]) - (count[startindex] - (cummcount[startindex] - binstart[bin] + 1)) - (cummcount[endindex] - binend[bin]) == bincount[bin]

        xbin[bin] = xsum/bincount[bin]
        ybin[bin] = ysum/bincount[bin]

        startindex = endindex
    end
    return xbin, ybin
end

function dataplot!(ax, groupeddf::DataFrame, ungroupeddf::DataFrame, result::String, numbins::Int64)::Nothing
    oddsimpliedbin, observedbin = binnify(Int64.(groupeddf.Count), Float64.(groupeddf.OddsimpliedProbresult), Float64.(groupeddf.ObservedProbresult), numbins)
    ungroupeddf.Diff = ungroupeddf.IsResult - ungroupeddf.OddsimpliedProbresult
    fit = lm(@formula(Diff ~ OddsimpliedProbresult + 1), ungroupeddf)  # , Bernoulli(), IdentityLink()
    println(fit)
    fit = lm(@formula(IsResult ~ OddsimpliedProbresult + 1), ungroupeddf)  # , Bernoulli(), IdentityLink()
    xs = [minimum(ungroupeddf.OddsimpliedProbresult):0.01:maximum(ungroupeddf.OddsimpliedProbresult);]
    pr = predict(fit, DataFrame(OddsimpliedProbresult = xs), interval = :confidence, level = 0.99)
    ys = pr.prediction
    intercept, slope = round.(coef(fit), RoundNearestTiesUp, digits = 2)
    # See: https://stackoverflow.com/questions/33060601/test-if-the-slope-in-simple-linear-regression-equals-to-a-given-constant-in-r
    # See: https://stats.stackexchange.com/questions/57492/what-test-should-be-used-to-tell-if-two-linear-regression-lines-are-significantl
    interceptsign = if (intercept < 0.0) L"$-$" else L"$+$" end
    interceptstring = @sprintf("%.2f", abs(intercept))
    slopestring = @sprintf("%.2f", slope)
    labelstring = "Observed = $(slopestring)" * L"\times" * "Odds-implied " * interceptsign * " $(interceptstring)"

    ax.scatter(oddsimpliedbin, observedbin, marker="o", color="dodgerblue", edgecolors="black", zorder=3)  # See: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    ax.plot(xs, ys, color="indianred", linewidth=2, label=labelstring, zorder=2)
    ax.plot([0.0, 1.0], [0.0, 1.0], color="darkgrey", linestyle="dashed", zorder=1)
    ax.set_xlabel("Odds-implied probability of $(result)")
    ax.set_ylabel("Observed probability of $(result)")
    ax.legend(loc="upper left", edgecolor="black", fancybox=false)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(true)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    return
end

# ==============================================================================
# Main program
# ==============================================================================

groupedwinquery = """
    SELECT
        Count(1) AS Count,
        OddsimpliedProbresult,
        AVG(IsResult) AS ObservedProbresult
    FROM (
        SELECT
            HomewinProb AS OddsimpliedProbresult,
            (CASE WHEN FulltimeHomeScore > FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
        FROM
            OddsImpliedProbs
        UNION ALL
        SELECT
            AwaywinProb AS OddsimpliedProbresult,
            (CASE WHEN FulltimeAwayScore > FulltimeHomeScore THEN 1.0 ELSE 0.0 END) AS IsResult
        FROM
            OddsImpliedProbs
        ) AS t
    GROUP BY OddsimpliedProbresult
    ORDER BY OddsimpliedProbresult
    """
groupedwindf = DBInterface.execute(db_conn, groupedwinquery) |> DataFrame
ungroupedwinquery = """
    (SELECT
        HomewinProb AS OddsimpliedProbresult,
        Convert(FLOAT, CASE WHEN FulltimeHomeScore > FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
    FROM
        OddsImpliedProbs
    )
    UNION ALL
    (SELECT
        AwaywinProb AS OddsimpliedProbresult,
        Convert(FLOAT, CASE WHEN FulltimeAwayScore > FulltimeHomeScore THEN 1.0 ELSE 0.0 END) AS IsResult
    FROM
        OddsImpliedProbs
    )
    """
ungroupedwindf = DBInterface.execute(db_conn, ungroupedwinquery) |> DataFrame
groupeddrawquery = """
    SELECT
        Count(1) AS Count,
        OddsimpliedProbresult,
        AVG(IsResult) AS ObservedProbresult
    FROM (
        SELECT
            1 - HomewinProb - AwaywinProb AS OddsimpliedProbresult,
            (CASE WHEN FulltimeHomeScore = FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
        FROM
            OddsImpliedProbs
        ) AS t
    GROUP BY OddsimpliedProbresult
    ORDER BY OddsimpliedProbresult
    """
groupeddrawdf = DBInterface.execute(db_conn, groupeddrawquery) |> DataFrame
ungroupeddrawquery = """
    (SELECT
        1 - HomewinProb - AwaywinProb AS OddsimpliedProbresult,
        Convert(FLOAT, CASE WHEN FulltimeHomeScore = FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
    FROM
        OddsImpliedProbs
    )
    """
ungroupeddrawdf = DBInterface.execute(db_conn, ungroupeddrawquery) |> DataFrame
groupedallquery = """
    SELECT
        Count(1) AS Count,
        OddsimpliedProbresult,
        AVG(IsResult) AS ObservedProbresult
    FROM (
        SELECT
            HomewinProb AS OddsimpliedProbresult,
            (CASE WHEN FulltimeHomeScore > FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
        FROM
            OddsImpliedProbs
        UNION ALL
        SELECT
            AwaywinProb AS OddsimpliedProbresult,
            (CASE WHEN FulltimeAwayScore > FulltimeHomeScore THEN 1.0 ELSE 0.0 END) AS IsResult
        FROM
            OddsImpliedProbs
        UNION ALL
        SELECT
            1 - HomewinProb - AwaywinProb AS OddsimpliedProbresult,
            (CASE WHEN FulltimeHomeScore = FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
        FROM
            OddsImpliedProbs
        ) AS t
    GROUP BY OddsimpliedProbresult
    ORDER BY OddsimpliedProbresult
    """
groupedalldf = DBInterface.execute(db_conn, groupedallquery) |> DataFrame
ungroupedallquery = """
    (SELECT
        HomewinProb AS OddsimpliedProbresult,
        Convert(FLOAT, CASE WHEN FulltimeHomeScore > FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
    FROM
        OddsImpliedProbs
    )
    UNION ALL
    (SELECT
        AwaywinProb AS OddsimpliedProbresult,
        Convert(FLOAT, CASE WHEN FulltimeAwayScore > FulltimeHomeScore THEN 1.0 ELSE 0.0 END) AS IsResult
    FROM
        OddsImpliedProbs
    )
    UNION ALL
    (SELECT
        1 - HomewinProb - AwaywinProb AS OddsimpliedProbresult,
        Convert(FLOAT, CASE WHEN FulltimeHomeScore = FulltimeAwayScore THEN 1.0 ELSE 0.0 END) AS IsResult
    FROM
        OddsImpliedProbs
    )

    """
ungroupedalldf = DBInterface.execute(db_conn, ungroupedallquery) |> DataFrame

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(5.2, 2.5))
dataplot!(ax1, groupedwindf, ungroupedwindf, "win", 30)
dataplot!(ax2, groupeddrawdf, ungroupeddrawdf, "draw", 15)
fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.NotPaper\\ObservedProbresultsVsOddsimpliedProbresults.WinDraw.png")

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(2.6, 2.5))
dataplot!(ax, groupedalldf, ungroupedalldf, "result", 30)
fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\ObservedProbresultsVsOddsimpliedProbresults.All.png")

# ==============================================================================
end
