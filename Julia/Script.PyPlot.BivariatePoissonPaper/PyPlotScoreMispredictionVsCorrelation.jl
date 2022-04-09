module PlotScoreMispredictionVsCorrelation
# ==============================================================================
#
# Modeling two football teams, A and B, with goal scoring distributed as:
# bivariate_poisson.pmf(goalsA, goalsB; lambdaA, lambdaB, lambdaX).
# ==============================================================================

using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/stable/
using OffsetArrays  # https://github.com/JuliaArrays/OffsetArrays.jl
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

MAX_GOALS = 5
MAX_Y = 20

# ==============================================================================
# Main program
# ==============================================================================

correlation = readcorrelations(db_conn)

homegoals, awaygoals = readmatchgoals(db_conn)
@assert length(homegoals) == length(awaygoals)
observedscorefrequency = observedscorefreq(homegoals, awaygoals, MAX_GOALS)

misprediction = Vector{Float64}(undef, length(correlation))
minmisprediction = Inf
minmispredictioncorr = NaN

c = 1
done = false
while !done
    corr = correlation[c]
    homemean, awaymean = readmatchimpliedmeans(db_conn, corr)
    @assert length(homemean) == length(awaymean) == length(homegoals) == length(awaygoals)
    predictedscorefrequency = predictedscorefreq(homemean, awaymean, corr, MAX_GOALS)
    weightedsquarederror = (predictedscorefrequency - observedscorefrequency).^2
    misprediction[c] = sqrt(sum(weightedsquarederror))*100
    println("corr = $(corr), misprediction = $(misprediction[c])")
    if misprediction[c] < minmisprediction
        global minmisprediction = misprediction[c]
        global minmispredictioncorr = corr
    end

    if c == length(correlation) || MAX_Y <= misprediction[c]
        global done = true
    else
        global c += 1
    end
end
println()
println("misprediction[$(minmispredictioncorr)] = $(minmisprediction)")
minmispredictioncorr2dp = round(minmispredictioncorr, RoundNearestTiesUp, digits=2)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 3))

#plt = plot(xlim = (0, 1.01), xticks = [0.0:0.1:1.0;], ylim = (0.0, MAX_Y), yticks = [5:5:MAX_Y;], legend = :topleft)
ax.plot(correlation[1:c], misprediction[1:c], color="blue", linestyle="solid", zorder=1)
ax.plot([minmispredictioncorr2dp, minmispredictioncorr2dp], [0, MAX_Y], color="grey", linestyle="dotted", label=L"$\rho_{A,B}$=" * @sprintf("%4.2f", minmispredictioncorr2dp), zorder=2)
ax.set_xlabel(L"$\rho_{A, B}$")
ax.set_ylabel("Root sum squared predicted" * L"$-$" * "observed score" * L"\%")
ax.legend(loc="upper left", edgecolor="black", fancybox=false)
ax.set_xlim(0, 1)
ax.set_yticks([0:5:MAX_Y;])
ax.set_ylim(0, MAX_Y)
ax.grid(true)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

fig.tight_layout(pad=0.25)  # See also: constrained_layout
display(plt.gcf())
plt.savefig("PyPlot.BivariatePoissonPaper\\ScoreMispredictionVsCorrelation.png")

# ==============================================================================
end
