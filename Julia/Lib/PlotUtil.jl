module PlotUtil
# ==============================================================================
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/stable/
using OffsetArrays  # https://github.com/JuliaArrays/OffsetArrays.jl
using Printf
using Distributions
using Plots
using Plots.PlotMeasures  # needed for px and mm notation in plots
using LaTeXStrings  # needed for L notation

using BivariatePoisson

export prettyprint
export readcorrelations, readmatchgoals, readmatchimpliedmeans
export observedscorefreq, predictedscorefreq
export plotscoreheatmap!, plotprobscoreheatmap, plotdiffscoreheatmap

# ==============================================================================
# Util functions
# ==============================================================================

function prettyprint(a::AbstractMatrix{Float64})::Nothing
    print("")
    print("    ")
    for j in axes(a, 2) @printf("%6i  ", j) end
    println()
    for i in axes(a, 1)
        @printf("%2i  ", i)
        for j in axes(a, 2) @printf("%6.2f  ", a[i, j]) end
        println()
    end
    return
end

# ==============================================================================
# Read functions
# ==============================================================================

function readcorrelations(db_conn::ODBC.Connection)::Vector{Float64}
    query = """
    SELECT DISTINCT
        Correlation
    FROM ProbsImpliedMeans
    WHERE Correlation IS NOT NULL
    ORDER BY Correlation
    """
    df = DBInterface.execute(db_conn, query) |> DataFrame
    return Float64.(df.Correlation)
end

function readmatchgoals(db_conn::ODBC.Connection)::Tuple{Vector{Int64}, Vector{Int64}}
    query = """
    SELECT
        FulltimeHomeScore,
        FulltimeAwayScore
    FROM
        OddsImpliedProbs
    """
    df = DBInterface.execute(db_conn, query) |> DataFrame
    return Int64.(df.FulltimeHomeScore), Int64.(df.FulltimeAwayScore)
end

function readmatchimpliedmeans(db_conn::ODBC.Connection, corr::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    query = """
    SELECT
        HomeGoalsMean,
        AwayGoalsMean
    FROM
        OddsImpliedProbs
    LEFT JOIN
        ProbsImpliedMeans
    ON ProbsImpliedMeans.AwaywinProb3DP = OddsImpliedProbs.AwaywinProb3DP AND ProbsImpliedMeans.HomewinProb3DP = OddsImpliedProbs.HomewinProb3DP
    WHERE Correlation = $(corr)
    """
    df = DBInterface.execute(db_conn, query) |> DataFrame
    return Float64.(df.HomeGoalsMean), Float64.(df.AwayGoalsMean)
end

# ==============================================================================
# Analysis functions
# ==============================================================================

function observedscorefreq(homegoals::Vector{Int64}, awaygoals::Vector{Int64}, maxgoals::Int64)::OffsetArray{Float64}
    @assert length(homegoals) == length(awaygoals)
    @assert 0 <= minimum(homegoals) && 0 <= minimum(awaygoals)
    @assert 0 <= maxgoals
    count = OffsetArray{Float64}(zeros(maxgoals + 2, maxgoals + 2), 0:(maxgoals + 1), 0:(maxgoals + 1))
    for (hg, ag) in zip(homegoals, awaygoals)
        (h, g) = min.((hg, ag), maxgoals + 1)
        count[h, g] += 1
    end
    return count/length(homegoals)
end

function predictedscorefreq(homemean::Vector{Float64}, awaymean::Vector{Float64}, corr::Float64, maxgoals::Int64)::OffsetArray{Float64}
    @assert length(homemean) == length(awaymean)
    @assert 0.0 <= minimum(homemean) && 0.0 <= minimum(awaymean)
    @assert 0 <= maxgoals
    cummulativeprob = OffsetArray{Float64}(zeros(maxgoals + 2, maxgoals + 2), 0:(maxgoals + 1), 0:(maxgoals + 1))
    for (hm, am) in zip(homemean, awaymean)
        for hg in 0:maxgoals
            for ag in 0:maxgoals
                prob = probscore_from_means(hg, ag, hm, am, corr)
                cummulativeprob[hg, ag] += prob
                cummulativeprob[hg, maxgoals + 1] -= prob
                cummulativeprob[maxgoals + 1, ag] -= prob
                cummulativeprob[maxgoals + 1, maxgoals + 1] += prob
            end
        end
        for g in 0:maxgoals
            hprob = pdf(Poisson(hm), g)
            aprob = pdf(Poisson(am), g)
            cummulativeprob[g, maxgoals + 1] += hprob
            cummulativeprob[maxgoals + 1, g] += aprob
            cummulativeprob[maxgoals + 1, maxgoals + 1] -= (hprob + aprob)
        end
        cummulativeprob[maxgoals + 1, maxgoals + 1] += 1.0
    end
    return cummulativeprob/length(homemean)
end

# ==============================================================================
# Plot functions
# ==============================================================================

function plotscoreheatmap!(ax, data::OffsetArray{Float64}, corr::Float64; title::Union{String, Nothing}, ylabel::Bool, clim::Tuple{Float64, Float64}, cmap)::Nothing
    @assert length(size(data)) == 2
    @assert axes(data, 1) == axes(data, 2)
    @assert first(axes(data, 1)) == 0 == first(axes(data, 2))
    @assert last(axes(data, 1)) == last(axes(data, 2))
    @assert 0.0 <= corr <= 1.0
    maxgoals = last(axes(data, 1)) - 1
    roundeddata = round.(transpose(data).*100, RoundNearestTiesUp, digits=1)
    goalsticks = 0:(maxgoals + 1)
    goallabels = vcat(string.(0:maxgoals), [">$(maxgoals)"])
    im = ax.imshow(roundeddata, aspect="equal", origin="lower", clim=clim, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Home Goals")
    if ylabel ax.set_ylabel("Away Goals"); ax.yaxis.set_label_coords(-0.12, 0.5) end
    ax.set_xlim(-0.5, maxgoals + 1.5)
    ax.set_xticks(goalsticks)
    ax.set_xticklabels(goallabels)
    ax.xaxis.set_tick_params(width=0)
    for label in ax.xaxis.get_ticklabels() label.set_y(+0.03) end
    ax.set_ylim(-0.5, maxgoals + 1.5)
    ax.set_yticks(goalsticks)
    ax.yaxis.set_tick_params(width=0)
    ax.set_yticklabels(goallabels)
    ax.yaxis.set_tick_params(width=0)
    for label in ax.yaxis.get_ticklabels() label.set_x(+0.03) end
    for i in axes(roundeddata, 1) .+ 0.5
        ax.plot([i, i], [-0.5, maxgoals + 1.5], color="grey", linewidth=1)
        ax.plot([-0.5, maxgoals + 1.5], [i, i], color="grey", linewidth=1)
    end
    for i in axes(roundeddata, 1) for j in axes(roundeddata, 2)
        text = @sprintf("%4.1f%%", roundeddata[i, j])
        im.axes.text(j, i, text, ha="center", fontsize=6, va="center", color="black")
    end end
    return
end

function plotscoreheatmap(probscore::OffsetArray{Float64}, corr::Float64; title::Union{String, Nothing}, ylabel::Bool, clim::Tuple{Float64, Float64}, colors::Vector{Symbol}, colorbar::Bool)::Plots.Plot
    @assert length(size(probscore)) == 2
    @assert axes(probscore, 1) == axes(probscore, 2)
    @assert first(axes(probscore, 1)) == 0 == first(axes(probscore, 2))
    @assert last(axes(probscore, 1)) == last(axes(probscore, 2))
    @assert 0.0 <= corr <= 1.0
    maxgoals = last(axes(probscore, 1)) - 1
    prob = round.(transpose(parent(probscore)).*100, RoundNearestTiesUp, digits=1)
    goalsticks = 1:(maxgoals + 2)
    goallabels = vcat(string.(0:maxgoals), [">$(maxgoals)"])
    plt = plot(title=title, xlabel="Home goals", ylabel=if ylabel "Away goals" else nothing end)
    plot!(left_margin=0px, right_margin=0px, top_margin=0px, bottom_margin=0px)
    plot!(titlefontsize=9, xtickfontsize=7, ytickfontsize=7, xguidefontsize=8, yguidefontsize=8)
    heatmap!(goallabels, goallabels, prob,
        xlim=(0, maxgoals + 2.05), ylim=(0, maxgoals + 2.05), clim=clim,
        aspect_ratio=:equal,
        color=cgrad(colors),
        colorbar=colorbar)
    annotation = [(j - 0.5, i - 0.5, text(@sprintf("%4.1f%%", prob[i, j]), 6, :black, :center))
                for i in axes(prob, 1) for j in axes(prob, 2)]
    for i in 1:maxgoals+2
        plot!([i, i], [0, maxgoals + 2], color = :grey, linewidth=1, label=nothing)
        plot!([0, maxgoals + 2], [i, i], color = :grey, linewidth=1, label=nothing)
    end
    annotate!(annotation, linecolor=:black)
    return plt
end

function plotdiffscoreheatmap(diffscore::OffsetArray{Float64}, corr::Float64; title::Union{String, Nothing}, ylabel::Bool, clim::Tuple{Float64, Float64}, colors::Vector{Symbol}, colorbar::Bool)::Plots.Plot
    @assert length(size(diffscore)) == 2
    @assert axes(diffscore, 1) == axes(diffscore, 2)
    @assert first(axes(diffscore, 1)) == 0 == first(axes(diffscore, 2))
    @assert last(axes(diffscore, 1)) == last(axes(diffscore, 2))
    @assert 0.0 <= corr <= 1.0
    maxgoals = last(axes(diffscore, 1)) - 1
    diff = round.(transpose(parent(diffscore)).*100, RoundNearestTiesUp, digits=1)
    goalsticks = 1:(maxgoals + 2)
    goallabels = vcat(string.(0:maxgoals), [">$(maxgoals)"])
    plt = plot(title=title, xlabel="Home goals", ylabel=if ylabel "Away goals" else nothing end)
    plot!(left_margin=0px, right_margin=0px, top_margin=0px, bottom_margin=0px)
    plot!(titlefontsize=9, xtickfontsize=7, ytickfontsize=7, xguidefontsize=8, yguidefontsize=8)
    heatmap!(goallabels, goallabels, diff,
        xlim=(0, maxgoals + 2.05), ylim=(0, maxgoals + 2.05), clim=clim,
        aspect_ratio=:equal,
        color=cgrad(colors),
        colorbar=colorbar)
    annotation = [(j - 0.5, i - 0.5, text(@sprintf("%4.1f%%", diff[i, j]), 6, :black, :center))
                for i in axes(diff, 1) for j in axes(diff, 2)]
    for i in 1:maxgoals+2
        plot!([i, i], [0, maxgoals + 2], color = :grey, linewidth=1, label=nothing)
        plot!([0, maxgoals + 2], [i, i], color = :grey, linewidth=1, label=nothing)
    end
    annotate!(annotation, linecolor=:black)
    return plt
end

# ==============================================================================
end
