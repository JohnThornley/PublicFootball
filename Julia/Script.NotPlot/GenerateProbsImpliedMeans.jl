module GenerateProbsImpliedMeans
# ==============================================================================
# Generate (mean(homegoals), mean(awaygoals)) from
#          (Pr(awaywin), Pr(homewin), correlation(homegoals, awaygoals))
# for all matches in Football..OddsImpliedProbs view and a range of correlation values
# and write to Football..ProbsImpliedMeans table in SQL Server database.
#
# INPUTS:
#
# - Football..OddsImpliedProbs view on (local) SQL Server
#
# OUTPUTS:
#
# - Football..ProbsImpliedMeans table on (local) SQL Server
#
# CONFIGURATION:
#
# - Correlations: 0.00, 0.01, 0.02, ... 0.20, 0.21, 0.22, ... 0.93, 0.94, 0.95
# - Numeric accuracy: 3 decimal places (chosen to match accuracy of input bookmaker odds)
#
# This script is restartable, i.e., it only generates missing values, not all values from scratch.
# ==============================================================================

using DecFP  # See: https://github.com/JuliaMath/DecFP.jl
using DataFrames  # See: https://dataframes.juliadata.org/stable/
using ODBC  # See: https://juliadatabases.github.io/ODBC.jl/v1.0/

using BivariatePoisson

# ==============================================================================
# Fixed values
# ==============================================================================

DP = 3
CORRELATIONS = 0.12:0.001:0.15 # 0.0:0.01:0.95
db_conn = ODBC.Connection("football", "football", "football2015")

# ==============================================================================
# Main program
# ==============================================================================

for correlation in CORRELATIONS
    local corr = round(Dec64(correlation), digits = DP)
    local query = """
        SELECT
            AwaywinProb, HomewinProb
        FROM (
            SELECT
                OddsImpliedProbs.AwaywinProb3DP AS AwaywinProb, OddsImpliedProbs.HomewinProb3DP AS HomewinProb, Correlation
            FROM
                OddsImpliedProbs LEFT JOIN ProbsImpliedMeans
            ON
                ProbsImpliedMeans.AwaywinProb3DP = OddsImpliedProbs.AwaywinProb3DP AND
                ProbsImpliedMeans.HomewinProb3DP = OddsImpliedProbs.HomewinProb3DP AND
                ProbsImpliedMeans.Correlation = $(corr)
            GROUP BY
                OddsImpliedProbs.AwaywinProb3DP, OddsImpliedProbs.HomewinProb3DP, Correlation
        ) AS t
        WHERE Correlation IS NULL
        """
    local df = DBInterface.execute(db_conn, query) |> DataFrame
    local numrows = nrow(df)
    if (0 < numrows)
        println("correlation = $(corr), numrows = $(numrows)")
        for (r, row) in enumerate(eachrow(df))
            BivariatePoisson.clear_caches!()
            awaywinprob = row.AwaywinProb
            homewinprob = row.HomewinProb
            homegoalsmean, awaygoalsmean = round.(Dec64.(means_from_probresults(Float64(awaywinprob), Float64(homewinprob), correlation, 10.0^(-DP))), digits = DP)
            DBInterface.execute(db_conn, """
                INSERT INTO ProbsImpliedMeans (Correlation, AwaywinProb3DP, HomewinProb3DP, HomeGoalsMean, AwayGoalsMean)
                VALUES ($(corr), $(awaywinprob), $(homewinprob), $(homegoalsmean), $(awaygoalsmean))
                """)
            if (r % 100 == 0) print(".") end
        end
        println()
    end
end
println("Done.")

# ==============================================================================
end
