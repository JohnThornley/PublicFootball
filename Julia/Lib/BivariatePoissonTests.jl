module BivariatePoissonTests
# ==============================================================================
# BivariatePoissonTests
#
# Unit tests for functions exported by BivariatePoisson.jl.
# ==============================================================================

using Test
using StatsBase
using Distributions

using BivariatePoisson

# ==============================================================================
# Constants
# ==============================================================================

ALMOST_EQUAL_DP = 14
ALMOST_ZERO = 10.0^(-ALMOST_EQUAL_DP)
ALMOST_INF_MEAN = 1000000.0

CUMM_ALMOST_EQUAL_DP = 11  # For derived results where errors can accumulate
CUMM_ALMOST_ZERO = 10.0^(-CUMM_ALMOST_EQUAL_DP)

MAX_Y = 4
Y_RANGE = 0:1:MAX_Y

MAX_LAMBDA = 4
LAMBDA_RANGE = vcat([0.0, 0.1, 0.5], [1:1:MAX_LAMBDA;])
LAMBDAX_RANGE = LAMBDA_RANGE

MAX_MEAN = 4
MEAN_RANGE = vcat([0.0, 0.1, 0.5], [1:1:MAX_MEAN;])

CORR_DIVISOR = 10
FULL_CORR_RANGE = [0:1:CORR_DIVISOR;]/CORR_DIVISOR
function MEANS_CORR_RANGE(mean1::Float64, mean2::Float64)
    maxcorr = maxcorr12(mean1, mean2)
    StatsBase.rle(vcat([0.0], maxcorr*([1:(CORR_DIVISOR - 1);]/CORR_DIVISOR), [maxcorr]))[1]
end

PROB_DIVISOR = 8  # Must be even so that 0.5 is included in the range
@assert PROB_DIVISOR % 2 == 0
PROB_RANGE = [0:1:PROB_DIVISOR;]/PROB_DIVISOR

# ==============================================================================
# test_probscore_from_lambdas_*()
# test_probscore_from_means_*()
# ==============================================================================

function test_probscore_from_lambdas_basic()::Nothing
    println("Start test_probscore_from_lambdas_basic")
    # Zero correlation, Y1 zero, Y2 zero
    @test probscore_from_lambdas(0, 0, 0.0, 0.0, 0.0) == 1.0
    @test probscore_from_lambdas(1, 0, 0.0, 0.0, 0.0) == 0.0
    @test probscore_from_lambdas(0, 1, 0.0, 0.0, 0.0) == 0.0
    @test probscore_from_lambdas(1, 1, 0.0, 0.0, 0.0) == 0.0

    # Zero correlation, Y1 non-zero, Y2 zero
    @test 0.0 < probscore_from_lambdas(0, 0, 1.0, 0.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 1.0, 0.0, 0.0) < 1.0
    @test probscore_from_lambdas(0, 1, 1.0, 0.0, 0.0) == 0.0
    @test probscore_from_lambdas(1, 1, 1.0, 0.0, 0.0) == 0.0

    # Zero correlation, Y1 zero, Y2 non-zero
    @test 0.0 < probscore_from_lambdas(0, 0, 0.0, 1.0, 0.0) < 1.0
    @test probscore_from_lambdas(1, 0, 0.0, 1.0, 0.0) == 0.0
    @test 0.0 < probscore_from_lambdas(0, 1, 0.0, 1.0, 0.0) < 1.0
    @test probscore_from_lambdas(1, 1, 0.0, 1.0, 0.0) == 0.0

    # Zero correlation, Y1 non-zero, Y2 non-zero, Y1 == Y2
    @test 0.0 < probscore_from_lambdas(0, 0, 1.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 1.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(0, 1, 1.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 1.0, 1.0, 0.0) < 1.0

    # Zero correlation, Y1 non-zero, Y2 non-zero, Y1 > Y2
    @test 0.0 < probscore_from_lambdas(0, 0, 2.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 2.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(0, 1, 2.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 2.0, 1.0, 0.0) < 1.0

    # Zero correlation, Y1 non-zero, Y2 non-zero, Y1 < Y2
    @test 0.0 < probscore_from_lambdas(0, 0, 1.0, 2.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 1.0, 2.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(0, 1, 1.0, 2.0, 0.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 1.0, 2.0, 0.0) < 1.0

    # Partial correlation, Y1 non-zero, Y2 non-zero, Y1 == Y2
    @test 0.0 < probscore_from_lambdas(0, 0, 1.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 1.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(0, 1, 1.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 1.0, 1.0, 1.0) < 1.0

    # Partial correlation, Y1 non-zero, Y2 non-zero, Y1 > Y2
    @test 0.0 < probscore_from_lambdas(0, 0, 1.0, 0.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 1.0, 0.0, 1.0) < 1.0
    @test probscore_from_lambdas(0, 1, 1.0, 0.0, 1.0) == 0.0
    @test 0.0 < probscore_from_lambdas(1, 1, 1.0, 0.0, 1.0) < 1.0

    @test 0.0 < probscore_from_lambdas(0, 0, 2.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 2.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(0, 1, 2.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 2.0, 1.0, 1.0) < 1.0

    # Partial correlation, Y1 non-zero, Y2 non-zero, Y1 < Y2
    @test 0.0 < probscore_from_lambdas(0, 0, 0.0, 1.0, 1.0) < 1.0
    @test probscore_from_lambdas(1, 0, 0.0, 1.0, 1.0) == 0.0
    @test 0.0 < probscore_from_lambdas(0, 1, 0.0, 1.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 0.0, 1.0, 1.0) < 1.0

    @test 0.0 < probscore_from_lambdas(0, 0, 1.0, 2.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 0, 1.0, 2.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(0, 1, 1.0, 2.0, 1.0) < 1.0
    @test 0.0 < probscore_from_lambdas(1, 1, 1.0, 2.0, 1.0) < 1.0

    # 100% correlation, Y1 non-zero, Y2 non-zero
    @test 0.0 < probscore_from_lambdas(0, 0, 0.0, 0.0, 1.0) < 1.0
    @test probscore_from_lambdas(1, 0, 0.0, 0.0, 1.0) == 0.0
    @test probscore_from_lambdas(0, 1, 0.0, 0.0, 1.0) == 0.0
    @test 0.0 < probscore_from_lambdas(1, 1, 0.0, 0.0, 1.0) < 1.0

    # Higher rate => lower prob of zero result
    @test 0.0 < probscore_from_lambdas(0, 0, 2.0, 0.0, 0.0) < probscore_from_lambdas(0, 0, 1.0, 0.0, 0.0) <  1.0
    @test 0.0 < probscore_from_lambdas(0, 0, 0.0, 2.0, 0.0) < probscore_from_lambdas(0, 0, 0.0, 1.0, 0.0) <  1.0
    @test 0.0 < probscore_from_lambdas(0, 0, 0.0, 0.0, 2.0) < probscore_from_lambdas(0, 0, 0.0, 0.0, 1.0) <  1.0
    println("End test_probscore_from_lambdas_basic")
end

function test_probscore_from_means_basic()::Nothing
    println("Start test_probscore_from_means_basic")
    # Zero correlation, Y1 zero, Y2 zero
    @test probscore_from_means(0, 0, 0.0, 0.0, 0.0) == 1.0
    @test probscore_from_means(1, 0, 0.0, 0.0, 0.0) == 0.0
    @test probscore_from_means(0, 1, 0.0, 0.0, 0.0) == 0.0
    @test probscore_from_means(1, 1, 0.0, 0.0, 0.0) == 0.0

    # Zero correlation, Y1 non-zero, Y2 zero
    @test 0.0 < probscore_from_means(0, 0, 1.0, 0.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 1.0, 0.0, 0.0) < 1.0
    @test probscore_from_means(0, 1, 1.0, 0.0, 0.0) == 0.0
    @test probscore_from_means(1, 1, 1.0, 0.0, 0.0) == 0.0

    # Zero correlation, Y1 zero, Y2 non-zero
    @test 0.0 < probscore_from_means(0, 0, 0.0, 1.0, 0.0) < 1.0
    @test probscore_from_means(1, 0, 0.0, 1.0, 0.0) == 0.0
    @test 0.0 < probscore_from_means(0, 1, 0.0, 1.0, 0.0) < 1.0
    @test probscore_from_means(1, 1, 0.0, 1.0, 0.0) == 0.0

    # Zero correlation, Y1 non-zero, Y2 non-zero, Y1 == Y2
    @test 0.0 < probscore_from_means(0, 0, 1.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 1.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(0, 1, 1.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 1, 1.0, 1.0, 0.0) < 1.0

    # Zero correlation, Y1 non-zero, Y2 non-zero, Y1 > Y2
    @test 0.0 < probscore_from_means(0, 0, 2.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 2.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(0, 1, 2.0, 1.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 1, 2.0, 1.0, 0.0) < 1.0

    # Zero correlation, Y1 non-zero, Y2 non-zero, Y1 > Y2
    @test 0.0 < probscore_from_means(0, 0, 1.0, 2.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 1.0, 2.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(0, 1, 1.0, 2.0, 0.0) < 1.0
    @test 0.0 < probscore_from_means(1, 1, 1.0, 2.0, 0.0) < 1.0

    # Y1 non-zero, Y2 non-zero, Y1 == Y2
    @test 0.0 < probscore_from_means(0, 0, 1.0, 1.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 1.0, 1.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(0, 1, 1.0, 1.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(1, 1, 1.0, 1.0, 0.5) < 1.0

    # # Partial correlation, Y1 non-zero, Y2 non-zero, Y1 > Y2
    @test 0.0 < probscore_from_means(0, 0, 2.0, 1.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 2.0, 1.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(0, 1, 2.0, 1.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(1, 1, 2.0, 1.0, 0.5) < 1.0

    # # Partial correlation, Y1 non-zero, Y2 non-zero, Y1 < Y2
    @test 0.0 < probscore_from_means(0, 0, 1.0, 2.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(1, 0, 1.0, 2.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(0, 1, 1.0, 2.0, 0.5) < 1.0
    @test 0.0 < probscore_from_means(1, 1, 1.0, 2.0, 0.5) < 1.0

    # 100% correlation, Y1 zero, Y2 zero
    @test probscore_from_means(0, 0, 0.0, 0.0, 1.0) == 1.0
    @test probscore_from_means(1, 0, 0.0, 0.0, 1.0) == 0.0
    @test probscore_from_means(0, 1, 0.0, 0.0, 1.0) == 0.0
    @test probscore_from_means(1, 1, 0.0, 0.0, 1.0) == 0.0

    # 100% correlation, Y1 non-zero, Y2 non-zero, Y1 == Y2
    @test 0.0 < probscore_from_means(0, 0, 1.0, 1.0, 1.0) < 1.0
    @test probscore_from_means(1, 0, 1.0, 1.0, 1.0) == 0.0
    @test probscore_from_means(0, 1, 1.0, 1.0, 1.0) == 0.0
    @test 0.0 < probscore_from_means(1, 1, 1.0, 1.0, 1.0) < 1.0

    # Higher rate => lower prob of zero result
    @test 0.0 < probscore_from_means(0, 0, 2.0, 0.0, 0.0) < probscore_from_means(0, 0, 1.0, 0.0, 0.0) <  1.0
    @test 0.0 < probscore_from_means(0, 0, 0.0, 2.0, 0.0) < probscore_from_means(0, 0, 0.0, 1.0, 0.0) <  1.0
    @test 0.0 < probscore_from_means(0, 0, 2.0, 2.0, 1.0) < probscore_from_means(0, 0, 1.0, 1.0, 1.0) <  1.0
    println("End test_probscore_from_means_basic")
end

function test_probscore_from_lambdas_sums()::Nothing
    println("Start test_probscore_from_lambdas_sums")
    # One distribution non-zero, other distribution zero: Sum of i-0 results should approach 1.0
    sum1 = 0.0
    sum4 = 0.0
    for i in 0:20
        prob1 = probscore_from_lambdas(i, 0, 1.0, 0.0, 0.0)
        prob2 = probscore_from_lambdas(0, i, 1.0, 0.0, 0.0)
        prob3 = probscore_from_lambdas(i, 0, 0.0, 1.0, 0.0)
        prob4 = probscore_from_lambdas(0, i, 0.0, 1.0, 0.0)
        @test 0.0 <= prob1 <= 1.0
        @test (prob2 == 0.0) == (i != 0)
        @test (prob3 == 0.0) == (i != 0)
        @test 0.0 <= prob4 <= 1.0
        sum1 += prob1
        sum4 += prob4
    end
    @test 1.0 - ALMOST_ZERO < sum1 <= 1.0
    @test 1.0 - ALMOST_ZERO < sum4 <= 1.0
    # 100% correlated distributions: Sum of i-1 results should approach 1.0
    sum = 0.0
    for i in 0:20
        for j in 0:20
            prob = probscore_from_lambdas(i, j, 0.0, 0.0, 1.0)
            @test (prob > 0.0) == (i == j)
            @test (prob == 0.0) == (i != j)
            sum += prob
    end end
    @test 0.99999999 < sum <= 1.0
    println("End test_probscore_from_lambdas_sums")
end

function test_probscore_from_means_sums()::Nothing
    println("Start test_probscore_from_means_sums")
    # One distribution non-zero, other distribution zero: Sum of i-0 results should approach 1.0
    sum1 = 0.0
    sum4 = 0.0
    for i in 0:20
        prob1 = probscore_from_means(i, 0, 1.0, 0.0, 0.0)
        prob2 = probscore_from_means(0, i, 1.0, 0.0, 0.0)
        prob3 = probscore_from_means(i, 0, 0.0, 1.0, 0.0)
        prob4 = probscore_from_means(0, i, 0.0, 1.0, 0.0)
        @test 0.0 <= prob1 <= 1.0
        @test (prob2 == 0.0) == (i != 0)
        @test (prob3 == 0.0) == (i != 0)
        @test 0.0 <= prob4 <= 1.0
        sum1 += prob1
        sum4 += prob4
    end
    @test 1.0 - ALMOST_ZERO < sum1 <= 1.0
    @test 1.0 - ALMOST_ZERO < sum4 <= 1.0
    # 100% correlated distributions: Sum of i-1 results should approach 1.0
    sum = 0.0
    for i in 0:20
        for j in 0:20
            prob = probscore_from_means(i, j, 1.0, 1.0, 1.0)
            @test (prob > 0.0) == (i == j)
            @test (prob == 0.0) == (i != j)
            sum += prob
    end end
    @test 1.0 - ALMOST_ZERO < sum <= 1.0
    println("End test_probscore_from_means_sums")
end

function test_probscore_from_lambdas_axioms()::Nothing
    println("Start test_probscore_from_lambdas_axioms")
    for y1 in Y_RANGE
        for y2 in Y_RANGE
            for lambda1 in LAMBDA_RANGE
                for lambda2 in LAMBDA_RANGE
                    for lambdax in LAMBDAX_RANGE
                        prob = probscore_from_lambdas(y1, y2, lambda1, lambda2, lambdax)
                        @test 0.0 <= prob <= 1.0
                        @test (prob < ALMOST_ZERO) == ((y1 > y2 && lambda1 == 0.0) || (y2 > y1 && lambda2 == 0.0) || (y1 > 0 && lambda1 == lambdax == 0.0) || (y2 > 0 && lambda2 == lambdax == 0.0))
                        @test (1.0 - prob < ALMOST_ZERO) == (y1 == y2 == 0 && lambda1 == lambda2 == lambdax == 0.0)
    end end end end end
    println("End test_probscore_from_lambdas_axioms")
end

function test_probscore_from_means_axioms()::Nothing
    println("Start test_probscore_from_means_axioms")
    for y1 in Y_RANGE
        for y2 in Y_RANGE
            for mean1 in MEAN_RANGE
                for mean2 in MEAN_RANGE
                    for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                        maxcorr = maxcorr12(mean1, mean2)
                        prob = probscore_from_means(y1, y2, mean1, mean2, corr12)
                        @test 0.0 <= prob <= 1.0
                        @test (prob < ALMOST_ZERO) == ((y2 < y1 && mean1 <= mean2 && corr12 == maxcorr) || (y1 < y2 && mean2 <= mean1 && corr12 == maxcorr) || (mean1 == 0.0 && 0 < y1) || (mean2 == 0 && 0 < y2))
                        @test (1.0 - prob < ALMOST_ZERO) == (y1 == y2 == 0 && mean1 == mean2 == 0.0)
    end end end end end
    println("End test_probscore_from_means_axioms")
end

function test_probscore_from_lambdas_symmetry()::Nothing
# prob(y1, y2, lambda1, lambda2, lambdax) == prob(y2, y1, lambda2, lambda1, lambdax)
    println("Start test_probscore_from_lambdas_symmetry")
    for y1 in Y_RANGE
        for y2 in Y_RANGE
            for lambda1 in LAMBDA_RANGE
                for lambda2 in LAMBDA_RANGE
                    for lambdax in LAMBDAX_RANGE
                        prob12 = probscore_from_lambdas(y1, y2, lambda1, lambda2, lambdax)
                        prob21 = probscore_from_lambdas(y2, y1, lambda2, lambda1, lambdax)
                        @test prob12 == prob21
    end end end end end
    println("End test_probscore_from_lambdas_symmetry")
end

function test_probscore_from_means_symmetry()::Nothing
# prob(y1, y2, mean1, mean2, corr12) == prob(y2, y1, mean2, mean1, corr12)
    println("Start test_probscore_from_means_symmetry")
    for y1 in Y_RANGE
        for y2 in Y_RANGE
            for mean1 in MEAN_RANGE
                for mean2 in MEAN_RANGE
                    for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                        prob12 = probscore_from_means(y1, y2, mean1, mean2, corr12)
                        prob21 = probscore_from_means(y2, y1, mean2, mean1, corr12)
                        @test prob12 == prob21
    end end end end end
    println("End test_probscore_from_means_symmetry")
end

function test_probscore_from_lambdas_double_poisson_equivalence()::Nothing
# prob(y1, y2; lambda1, lambda2, lambdax = 0.0) == poisson.pdf(y1, lambda1)*poisson.pdf(y2, lambda2)
    println("Start test_probscore_from_lambdas_double_poisson_equivalence")
    for y1 in Y_RANGE
        for y2 in Y_RANGE
            for lambda1 in LAMBDA_RANGE
                poissonprob1 = pdf(Poisson(lambda1), y1)
                for lambda2 in LAMBDA_RANGE
                    poissonprob2 = pdf(Poisson(lambda2), y2)
                    bivariate_poission_prob = probscore_from_lambdas(y1, y2, lambda1, lambda2, 0.0)
                    double_poisson_prob = poissonprob1*poissonprob2
                    @test abs(bivariate_poission_prob - double_poisson_prob) < ALMOST_ZERO
    end end end end
    println("End test_probscore_from_lambdas_double_poisson_equivalence")
end

function test_probscore_from_means_double_poisson_equivalence()::Nothing
# prob(y1, y2; mean1, mean2, corr12 = 0.0) == poisson.pdf(y1, mean1)*poisson.pdf(y2, mean2)
    println("Start test_probscore_from_means_double_poisson_equivalence")
    for y1 in Y_RANGE
        for y2 in Y_RANGE
            for mean1 in MEAN_RANGE
                poissonprob1 = pdf(Poisson(mean1), y1)
                for mean2 in MEAN_RANGE
                    poissonprob2 = pdf(Poisson(mean2), y2)
                    bivariate_poission_prob = probscore_from_lambdas(y1, y2, mean1, mean2, 0.0)
                    double_poisson_prob = poissonprob1*poissonprob2
                    @test abs(bivariate_poission_prob - double_poisson_prob) < ALMOST_ZERO
    end end end end
    println("End test_probscore_from_means_double_poisson_equivalence")
end

function test_marginal_poisson(marginalmean1::Float64, marginalmean2::Float64, bivariatepoissonpdf)::Nothing
# sum(bivariatepoissonpdf(y, *)) is distributed Poisson(marginalmean1).pdf(y)
# sum(bivariatepoissonpdf(*, y)) is distributed Poisson(marginalmean2).pdf(y)
    for y in Y_RANGE
        marginalprob1 = 0.0
        marginalprob2 = 0.0
        prev_marginalprob1 = -1.0
        prev_marginalprob2 = -1.0
        converged = false
        z = 0
        while !converged # for z in range (0..) while not converged
            marginalprob1 += bivariatepoissonpdf(y, z)
            marginalprob2 += bivariatepoissonpdf(z, y)
            converged = z > y && marginalprob1 == prev_marginalprob1 && marginalprob2 == prev_marginalprob2
            prev_marginalprob1 = marginalprob1
            prev_marginalprob2 = marginalprob2
            z += 1
        end
        poissonprob1 = pdf(Poisson(marginalmean1), y)
        @test abs(poissonprob1 - marginalprob1) < CUMM_ALMOST_ZERO
        poissonprob2 = pdf(Poisson(marginalmean2), y)
        @test abs(poissonprob2 - marginalprob2) < CUMM_ALMOST_ZERO
    end
end

function test_probscore_from_lambdas_marginal_poisson()::Nothing
# For any lambda1, lambda2, lambdax: sum(prob(y, *, lambda1, lambda2, lambdax)) is distributed poisson.pdf(y, lambda1 + lambdax)
# For any lambda1, lambda2, lambdax: sum(prob(*, y, lambda1, lambda2, lambdax)) is distributed poisson.pdf(y, lambda2 + lambdax)
    println("Start test_probscore_from_lambdas_marginal_poisson")
    for lambda1 in LAMBDA_RANGE
        for lambda2 in LAMBDA_RANGE
            for lambdax in LAMBDAX_RANGE
                function bivariatepoissonpdf(y1::Int64, y2::Int64)::Float64 return probscore_from_lambdas(y1, y2, lambda1, lambda2, lambdax) end
                test_marginal_poisson(lambda1 + lambdax, lambda2 + lambdax, bivariatepoissonpdf)
    end end end
    println("End test_probscore_from_lambdas_marginal_poisson")
end

function test_probscore_from_means_marginal_poisson()::Nothing
# For any mean1, mean2, corr12: sum(prob(y, *, mean1, mean2, corr12)) is distributed poisson.pmf(y, mean1)
# For any mean1, mean2, corr12: sum(prob(*, y, mean1, mean2, corr12)) is distributed poisson.pmf(y, mean2)
    println("Start test_probscore_from_means_marginal_poisson")
    for mean1 in MEAN_RANGE
        for mean2 in MEAN_RANGE
            for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                function bivariatepoissonpdf(y1::Int64, y2::Int64)::Float64 return probscore_from_means(y1, y2, mean1, mean2, corr12) end
                test_marginal_poisson(mean1, mean2, bivariatepoissonpdf)
    end end end
    println("End test_probscore_from_means_marginal_poisson")
end

function test_probscore_cummulative_stats(bivariatepoissonpdf, expected_mean1::Float64, expected_mean2::Float64, expected_var1::Float64, expected_var2::Float64, expected_cov12::Float64)::Nothing
# For a bivariate Poisson pmf(y1, y2) function with fixed parameterization,
# sum over y1, y2 in (0,0), (1,0), (0,1), (1,1), (2,0), (0,2), (2,1), (1,1), (2,2), ... until converged and check:
# sum(pmf)=1.0, mean(y1)=mean1, mean(y2)=mean1, var(y1)=var1, var(y2)=var2, and cov(y1,y2)=cov12.

    cummprob = 0.0
    prev_cummprob = -1.0
    # Mean y1
    mean1 = 0.0
    prev_mean1 = -1.0
    # Mean y1**2 for var(y1)
    meansq1 = 0.0
    var1 = 0.0
    prev_var1 = -1.0
    # Mean y2
    mean2 = 0.0
    prev_mean2 = -1.0
    # Mean y2**2 for var(y2)
    meansq2 = 0.0
    var2 = 0.0
    prev_var2 = -1.0
    # Mean y1*y2 for cov(y1, y2)
    mean12 = 0.0
    cov12 = 0.0
    prev_cov12 = -1.0

    function accumulate(y1::Int64, y2::Int64, bivariatepoissonpdf)
        @assert 0 <= y1 && 0 <= y2
        prob = bivariatepoissonpdf(y1, y2)
        @assert 0.0 <= prob <= 1.0
        return cummprob + prob, mean1 + y1*prob, meansq1 + y1*y1*prob, mean2 + y2*prob, meansq2 + y2*y2*prob, mean12 + y1*y2*prob
    end

    converged = false
    ya = 0
    while !converged  # for ya in range(0..) while not converged
        cummprob, mean1, meansq1, mean2, meansq2, mean12 = accumulate(ya, ya, bivariatepoissonpdf)
        for yb in 0:(ya - 1)
            cummprob, mean1, meansq1, mean2, meansq2, mean12 = accumulate(ya, yb, bivariatepoissonpdf)
            cummprob, mean1, meansq1, mean2, meansq2, mean12 = accumulate(yb, ya, bivariatepoissonpdf)
        end
        var1 = meansq1 - mean1*mean1
        var2 = meansq2 - mean2*mean2
        cov12 = mean12 - mean1*mean2
        converged =
            abs(prev_cummprob - cummprob) < CUMM_ALMOST_ZERO &&
            abs(prev_mean1 - mean1) < CUMM_ALMOST_ZERO &&
            abs(prev_mean2 - mean2) < CUMM_ALMOST_ZERO &&
            abs(prev_var1 - var1) < CUMM_ALMOST_ZERO &&
            abs(prev_var2 - var2) < CUMM_ALMOST_ZERO &&
            abs(prev_cov12 - cov12) < CUMM_ALMOST_ZERO
        prev_cummprob = cummprob
        prev_mean1 = mean1
        prev_mean2 = mean2
        prev_var1 = var1
        prev_var2 = var2
        prev_cov12 = cov12
        ya += 1
    end

    @test abs(cummprob - 1.0) < CUMM_ALMOST_ZERO
    @test abs(mean1 - expected_mean1) < CUMM_ALMOST_ZERO
    @test abs(mean2 - expected_mean2) < CUMM_ALMOST_ZERO
    @test abs(var1 - expected_var1) < CUMM_ALMOST_ZERO
    @test abs(var2 - expected_var2) < CUMM_ALMOST_ZERO
    @test abs(cov12 - expected_cov12) < CUMM_ALMOST_ZERO
    return
end

function test_probscore_from_lambdas_cummulative_stats()::Nothing
# For any lambda1, lambda2, lambdax:
# 1. sum(y1, y2: prob(y1, y2; lambda1, lambda2, lambdax) == 1.0
# 2. mean(y1) == lambda1 + lambdax and mean(y2) = lambda2 + lambdax
# 3. var(y1) = lambda1 + lambdax and var(y2) = lambda2 + lambdax
# 4. cov(y1, y2) = lambdax
    println("Start test_probscore_from_lambdas_cummulative_stats")
    for lambda1 in LAMBDA_RANGE
        for lambda2 in LAMBDA_RANGE
            for lambdax in LAMBDAX_RANGE
                function bivariatepoissonpdf(y1::Int64, y2::Int64)::Float64 return probscore_from_lambdas(y1, y2, lambda1, lambda2, lambdax) end
                test_probscore_cummulative_stats(bivariatepoissonpdf, lambda1 + lambdax, lambda2 + lambdax, lambda1 + lambdax, lambda2 + lambdax, lambdax)
    end end end
    println("End test_probscore_from_lambdas_cummulative_stats")
end

function test_probscore_from_means_cummulative_stats()::Nothing
# For any mean1, mean2, corr12:
# 1. sum(y1, y2: prob(y1, y2; mean1, mean2, corr12) == 1.0
# 2. mean(y1) == mean1 and mean(y2) = mean2
# 3. var(y1) = mean1 and var(y2) = mean2
# 4. cov(y1, y2) = corr12/sqrt(mean1*mean2)
    println("Start test_probscore_from_means_cummulative_stats")
    for mean1 in MEAN_RANGE
        for mean2 in MEAN_RANGE
            for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                function bivariatepoissonpdf(y1::Int64, y2::Int64)::Float64 return probscore_from_means(y1, y2, mean1, mean2, corr12) end
                test_probscore_cummulative_stats(bivariatepoissonpdf, mean1, mean2, mean1, mean2, corr12*sqrt(mean1*mean2))
    end end end
    println("End test_probscore_from_means_cummulative_stats")
end

# ==============================================================================
# test lambdas <-> means arithmetic
# ==============================================================================

function test_meansvalid_basic()::Nothing
    println("Start test_meansvalid_basic")
    @test meansvalid(0.0, 0.0, 0.0)
    @test meansvalid(0.0, 0.0, 0.5)
    @test meansvalid(0.0, 0.0, 1.0)
    @test meansvalid(1.0, 0.0, 0.0)
    @test !meansvalid(1.0, 0.0, 0.5)
    @test !meansvalid(1.0, 0.0, 1.0)
    @test meansvalid(0.0, 1.0, 0.0)
    @test !meansvalid(0.0, 1.0, 0.5)
    @test !meansvalid(0.0, 1.0, 1.0)
    @test meansvalid(1.0, 1.0, 0.0)
    @test meansvalid(1.0, 1.0, 0.5)
    @test meansvalid(1.0, 1.0, 1.0)
    @test meansvalid(1.0, 4.0, 0.0)
    @test meansvalid(1.0, 4.0, 0.5 - eps(0.5))
    @test !meansvalid(1.0, 4.0, 0.5 + eps(0.5))
    @test !meansvalid(1.0, 4.0, 1.0)
    @test meansvalid(4.0, 1.0, 0.0)
    @test meansvalid(4.0, 1.0, 0.5 - eps(0.5))
    @test !meansvalid(4.0, 1.0, 0.5 + eps(0.5))
    @test !meansvalid(4.0, 1.0, 1.0)
    println("End test_meansvalid_basic")
end

function test_minsmallermean_basic()::Nothing
    println("Start test_minsmallermean_basic")
    @test minsmallermean(0.0, 0.0) == 0.0
    @test minsmallermean(0.0, 0.5) == 0.0
    @test minsmallermean(0.0, 1.0) == 0.0
    @test minsmallermean(1.0, 0.0) == 0.0
    @test abs(minsmallermean(1.0, 0.5) - 0.25) <= eps(0.25)
    @test minsmallermean(1.0, 1.0) == 1.0
    println("End test_minsmallermean_basic")
end

function test_maxlargermean_basic()::Nothing
    println("Start test_maxlargermean_basic")
    @test isinf(maxlargermean(0.0, 0.0))
    @test maxlargermean(0.0, 0.5) == 0.0
    @test maxlargermean(0.0, 1.0) == 0.0
    @test isinf(maxlargermean(1.0, 0.0))
    @test maxlargermean(1.0, 0.5) <= 4.0 + eps(2.0)
    @test maxlargermean(1.0, 1.0) == 1.0
    println("End test_maxlargermean_basic")
end

function test_maxcorr12_basic()::Nothing
    println("Start test_maxcorr12_basic")
    @test maxcorr12(0.0, 0.0) == 1.0
    @test maxcorr12(1.0, 0.0) == 0.0
    @test maxcorr12(0.0, 1.0) == 0.0
    @test maxcorr12(1.0, 1.0) == 1.0
    @test maxcorr12(4.0, 4.0) == 1.0
    @test abs(maxcorr12(1.0, 4.0) - 0.5) <= eps(0.5)
    @test abs(maxcorr12(4.0, 1.0) - 0.5) <= eps(0.5)
    println("End test_maxcorr12_basic")
end

function test_means_axioms()::Nothing
    println("Start test_means_axioms")
    for mean1 in MEAN_RANGE
        for mean2 in MEAN_RANGE
            for corr12 in FULL_CORR_RANGE
                meansvalid_ = meansvalid(mean1, mean2, corr12)
                smallermean = min(mean1, mean2)
                largermean = max(mean1, mean2)
                minsmallermean_ = minsmallermean(largermean, corr12)
                maxlargermean_ = maxlargermean(smallermean, corr12)
                maxcorr12_ = maxcorr12(mean1, mean2)
                @test (isinf(maxlargermean_) && meansvalid(smallermean, ALMOST_INF_MEAN, corr12)) || meansvalid(smallermean, maxlargermean_, corr12)
                @test meansvalid(minsmallermean_, largermean, corr12)
                @test meansvalid_ == (minsmallermean_ <= smallermean + eps(smallermean))
                @test meansvalid_ == (isinf(maxlargermean_) || (largermean - eps(largermean) <= maxlargermean_))
                @test meansvalid_ == (corr12 <= maxcorr12_ + eps(maxcorr12_))
    end end end
    println("End test_means_axioms")
end

function test_means_range()::Nothing
    println("Start test_means_range")
    for mean in MEAN_RANGE
        for corr12 in FULL_CORR_RANGE
            minsmallermean_ = minsmallermean(mean, corr12)
            maxlargermean_ = maxlargermean(mean, corr12)
            @test meansvalid(minsmallermean_, mean, corr12)
            @test (isinf(maxlargermean_) && meansvalid(mean, ALMOST_INF_MEAN, corr12)) || meansvalid(mean, maxlargermean_, corr12)
            @test minsmallermean_ == 0.0 || !meansvalid(minsmallermean_/2, mean, corr12)
            @test isinf(maxlargermean_) || !meansvalid(mean, maxlargermean_ + 1.0, corr12)
    end end
    println("End test_means_range")
end

# test_probresults_from_lambdas_*()
# test_probresults_from_means_*()
# ==============================================================================

function test_probresults_from_lambdas_basic()::Nothing
    println("Start test_probresults_from_lambdas_basic")
    # lambdas equal, zero
    @test probresults_from_lambdas(0.0, 0.0, 0.0) == (0.0, 0.0)
    @test probresults_from_lambdas(0.0, 0.0, 1.0) == (0.0, 0.0)

    # lambdas equal, non-zero
    prob_x0, probequal = probresults_from_lambdas(1.0, 1.0, 0.0)
    @test 0.0 < prob_x0 == probequal < 0.5
    prob_x1, probequal = probresults_from_lambdas(1.0, 1.0, 1.0)
    @test 0.0 < prob_x1 == probequal < 0.5
    @test abs(prob_x0 - prob_x1) <= eps(2.0)

    # one lambda zero, other lambda non-zero
    prob_x0, zero = probresults_from_lambdas(0.0, 1.0, 0.0)
    @test zero == 0.0 < prob_x0 < 1.0
    @test probresults_from_lambdas(1.0, 0.0, 0.0) == (zero, prob_x0)
    prob_x1, zero = probresults_from_lambdas(0.0, 1.0, 1.0)
    @test zero == 0.0 < prob_x1 < 1.0
    @test probresults_from_lambdas(1.0, 0.0, 1.0) == (zero, prob_x1)
    @test abs(prob_x0 - prob_x1) <= eps(2.0)

    # lambdas non-equal, non-zero
    prob1_x0, prob2_x0 = probresults_from_lambdas(1.0, 2.0, 0.0)
    @test 0.0 < prob2_x0 < prob1_x0 < 1.0
    @test prob2_x0 + prob1_x0 < 1.0
    probresults_from_lambdas(2.0, 1.0, 0.0) == (prob2_x0, prob1_x0)
    prob1_x1, prob2_x1 = probresults_from_lambdas(1.0, 2.0, 1.0)
    probresults_from_lambdas(2.0, 1.0, 1.0) == (prob2_x1, prob1_x1)
    @test abs(prob1_x0 - prob1_x1) <= eps(2.0) && abs(prob2_x0 - prob2_x1) <= eps(2.0)

    # larger lambda => lower prob of zero result
    @test probresults_from_lambdas(0.0, 1.0, 0.0)[1] < probresults_from_lambdas(0.0, 2.0, 0.0)[1]
    @test probresults_from_lambdas(0.0, 1.0, 0.5)[1] < probresults_from_lambdas(0.0, 2.0, 0.5)[1]
    println("End test_probresults_from_lambdas_basic")
end

function test_probresults_from_means_basic()::Nothing
    println("Start test_probresults_from_means_basic")
    # means equal, zero
    @test probresults_from_means(0.0, 0.0, 0.0) == (0.0, 0.0)
    @test probresults_from_means(0.0, 0.0, 0.5) == (0.0, 0.0)
    @test probresults_from_means(0.0, 0.0, 1.0) == (0.0, 0.0)

    # means equal, non-zero
    prob_c0, probequal = probresults_from_means(1.0, 1.0, 0.0)
    @test 0.0 < prob_c0 == probequal < 0.5
    prob_c1, probequal = probresults_from_means(1.0, 1.0, 0.5)
    @test 0.0 < prob_c1 == probequal < prob_c0
    @test probresults_from_means(1.0, 1.0, 1.0) == (0.0, 0.0)

    # one mean zero, other mean non-zero
    prob_c0, zero = probresults_from_means(0.0, 1.0, 0.0)
    @test zero == 0.0 < prob_c0 < 1.0
    probresults_from_means(0.0, 1.0, 0.0) == (zero, prob_c0)

    # means non-equal, non-zero
    prob1_c0, prob2_c0 = probresults_from_means(1.0, 2.0, 0.0)
    @test 0.0 < prob2_c0 < prob1_c0 < 1.0
    @test prob2_c0 + prob1_c0 < 1.0
    probresults_from_means(2.0, 1.0, 0.0) == (prob2_c0, prob1_c0)
    prob1_c1, prob2_c1 = probresults_from_means(1.0, 2.0, 0.5)
    @test 0.0 < prob2_c1 < prob1_c1 < 1.0
    @test prob2_c0 + prob1_c1 < 1.0
    probresults_from_means(2.0, 1.0, 0.5) == (prob2_c1, prob1_c1)

    # larger mean => lower prob of zero result
    @test probresults_from_means(0.0, 1.0, 0.0)[1] < probresults_from_means(0.0, 2.0, 0.0)[1]
    println("End test_probresults_from_means_basic")
end

function test_probresults_from_lambdas_axioms()::Nothing
    println("Start test_probresults_from_lambdas_axioms")
    # 0.0 and 1.0 probabilities
    for lambda1 in LAMBDA_RANGE
        for lambda2 in LAMBDA_RANGE
            proby1lty2, proby1gty2 = probresults_from_lambdas(lambda1, lambda2, 0.0)
            proby1eqy2 = 1.0 - (proby1lty2 + proby1gty2)
            @test 0.0 <= proby1lty2 < 1.0
            @test 0.0 < proby1eqy2 <= 1.0
            @test 0.0 <= proby1gty2 < 1.0
            @test (proby1lty2 == 0.0) == (lambda2 == 0.0)
            @test (proby1gty2 == 0.0) == (lambda1 == 0.0)
            @test (proby1eqy2 == 1.0) == (lambda1 == lambda2 == 0.0)
            for lambdax in LAMBDAX_RANGE[2:length(LAMBDAX_RANGE)]
                proby1lty2x, proby1gty2x = probresults_from_lambdas(lambda1, lambda2, lambdax)
                @test abs(proby1lty2 - proby1lty2x) <= eps(8.0) && abs(proby1gty2 - proby1gty2x) <= eps(8.0)
    end end end
    println("End test_probresults_from_lambdas_axioms")
end

function test_probresults_from_means_axioms()::Nothing
# 0.0 and 1.0 probabilities
    println("Start test_probresults_from_means_axioms")
    for mean1 in MEAN_RANGE
        for mean2 in MEAN_RANGE
            for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                maxcorr = maxcorr12(mean1, mean2)
                proby1lty2, proby1gty2 = probresults_from_means(mean1, mean2, corr12)
                proby1eqy2 = 1.0 - (proby1lty2 + proby1gty2)
                @test 0.0 <= proby1lty2 < 1.0
                @test 0.0 < proby1eqy2 <= 1.0
                @test 0.0 <= proby1gty2 < 1.0
                @test (proby1lty2 <= ALMOST_ZERO) == ((mean1 == mean2 == 0) || (mean2 <= mean1 && corr12 == maxcorr))
                @test (proby1gty2 <= ALMOST_ZERO) == ((mean1 == mean2 == 0) || (mean1 <= mean2 && corr12 == maxcorr))
                @test (proby1eqy2 == 1.0) == ((mean1 == mean2 == 0) || (mean1 == mean2 && corr12 == maxcorr))
    end end end
    println("End test_probresults_from_means_axioms")
end

function test_probresults_from_lambdas_relations()::Nothing
# probresults(y1 < y2, lambda1, lambda1, lambdax) == probresults(y1 > y2, lambda1, lambda1, lambdax)
# probresults(lambda1, lambda1, lambdax) < probresults(lambda1 + delta, lambda1 + delta, lambdax)
# probresults(lambda1, lambda2, lambdax) == reverse(probresults(lambda1 + delta, lambda1 + delta, lambdax))
# prob(y1 < y2, lambda1, lambda2, lambdax) < probresults(y1 < y2, lambda1, lambda2 + delta, lambdax)
# prob(y1 > y2, lambda1, lambda2, lambdax) > probresults(y1 > y2, lambda1, lambda2 + delta, lambdax)
# prob(y1 < y2, lambda1, lambda2, lambdax) > probresults(y1 < y2, lambda1 + delta, lambda2, lambdax)
# prob(y1 > y2, lambda1, lambda2, lambdax) < probresults(y1 > y2, lambda1 + delta, lambda2, lambdax)
# Don't need to iterate over lambdax because lambdax doesn't affect probresults.
    println("Start test_probresults_from_lambdas_equal12")
    prev_prob_11 = -eps(1.0)
    for lambda1 in LAMBDA_RANGE
        proby1lty2_11, proby1gty2_11 = probresults_from_lambdas(lambda1, lambda1, 0.0)
        # probresults(y1 < y2, lambda1, lambda1, lambdax) == probresults(y1 > y2, lambda1, lambda1, lambdax)
        @test proby1lty2_11 == proby1gty2_11
        # probresults(lambda1, lambda1, lambdax) < probresults(lambda1 + delta, lambda1 + delta, lambdax)
        @test prev_prob_11 < proby1lty2_11
        prev_proby1lty2_12, prev_proby1gty2_12 = -eps(1.0), 1.0 + eps(1.0)
        prev_proby1lty2_21, prev_proby1gty2_21 = 1.0 + eps(1.0), -eps(1.0)
        for lambda2 in LAMBDA_RANGE
            proby1lty2_12, proby1gty2_12 = probresults_from_lambdas(lambda1, lambda2, 0.0)
            proby1lty2_21, proby1gty2_21 = probresults_from_lambdas(lambda2, lambda1, 0.0)
            # probresults(lambda1, lambda2, lambdax) == reverse(probresults(lambda1 + delta, lambda1 + delta, lambdax))
            @test (proby1lty2_12, proby1gty2_12) == (proby1gty2_21, proby1lty2_21)
            # prob(y1 < y2, lambda1, lambda2, lambdax) < probresults(y1 < y2, lambda1, lambda2 + delta, lambdax)
            @test prev_proby1lty2_12 < proby1lty2_12
            # prob(y1 > y2, lambda1, lambda2, lambdax) > probresults(y1 > y2, lambda1, lambda2 + delta, lambdax)
            @test (lambda1 == 0 && proby1gty2_12 == 0.0) || (lambda1 > 0.0 && prev_proby1gty2_12 > proby1gty2_12)
            # prob(y1 < y2, lambda1, lambda2, lambdax) > probresults(y1 < y2, lambda1 + delta, lambda2, lambdax)
            @test prev_proby1gty2_21 < proby1gty2_21
            # prob(y1 > y2, lambda1, lambda2, lambdax) < probresults(y1 > y2, lambda1 + delta, lambda2, lambdax)
            @test (lambda1 == 0 && proby1lty2_21 == 0.0) || (lambda1 > 0.0 && prev_proby1lty2_21 > proby1lty2_21)
            prev_proby1lty2_12, prev_proby1gty2_12 = proby1lty2_12, proby1gty2_12
            prev_proby1lty2_21, prev_proby1gty2_21 = proby1lty2_21, proby1gty2_21
    end end
    println("End test_probresults_from_lambdas_equal12")
end

function test_probresults_from_means_equal()::Nothing
# probresults(mean, mean, corr12) < probresults(mean + delta, mean + delta, corr12)
    println("Start test_probresults_from_means_equal12")
    for corr12 in FULL_CORR_RANGE
        prev_prob = -1.0
        for mean in MEAN_RANGE
            proby1lty2, proby1gty2 = probresults_from_means(mean, mean, corr12)
            @test (corr12 == 1.0 && proby1lty2 == proby1gty2 == 0.0) || (corr12 < 1.0 && prev_prob < proby1lty2 == proby1gty2)
            prev_prob = proby1lty2
    end end
    println("End test_probresults_from_means_equal12")
end

function test_probresults_from_means_symmetry()::Nothing
# probresults(mean1, mean2, corr12) == reverse(probresults(mean2, mean1, corr12))
    println("Start test_probresults_from_means_symmetry")
    for mean1 in MEAN_RANGE
        for mean2 in MEAN_RANGE
            for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                @test probresults_from_means(mean1, mean2, corr12) ==  reverse(probresults_from_means(mean2, mean1, corr12))
    end end end
    println("End test_probresults_from_means_symmetry")
end

function test_probresults_from_lambdas_double_poisson_equivalence()::Nothing
# prob(y1 < y2, lambda1, lambda2, lambdax = 0.0) == sum(y1 in 0:Inf, y2 in 0:Inf, where y1 < y2: poisson.pdf(y1, lambda1)*poisson.pdf(y2, lambda2))
# prob(y1 > y2, lambda1, lambda2, lambdax = 0.0) == sum(y1 in 0:Inf, y2 in 0:Inf, where y1 > y2: : poisson.pdf(y1, lambda1)*poisson.pdf(y2, lambda2))
    println("Start test_probresults_from_lambdas_double_poisson_equivalence")
    for lambda1 in LAMBDA_RANGE
        for lambda2 in LAMBDA_RANGE
            proby1lty2 = 0.0
            proby1eqy2 = 0.0
            proby1gty2 = 0.0
            probtotal = 0.0
            prev_probtotal = -1.0
            converged = false
            y1 = 0
            while !converged  # for y1 in range(0..) while not converged
                proby1eqy2 += pdf(Poisson(lambda1), y1)*pdf(Poisson(lambda2), y1)
                for y2 in 0:1:(y1 - 1)
                    proby1lty2 += pdf(Poisson(lambda1), y2)*pdf(Poisson(lambda2), y1)
                    proby1gty2 += pdf(Poisson(lambda1), y1)*pdf(Poisson(lambda2), y2)
                end
                probtotal = proby1lty2 + proby1eqy2 + proby1gty2
                converged = (probtotal == prev_probtotal)
                prev_probtotal = probtotal
                y1 += 1
            end
            expected_proby1lty2, expected_proby1gty2 = proby1lty2/probtotal, proby1gty2/probtotal
            computed_proby1lty2, computed_proby1gty2 = probresults_from_lambdas(lambda1, lambda2, 0.0)
            @test abs(expected_proby1lty2 - computed_proby1lty2) < ALMOST_ZERO
            @test abs(expected_proby1gty2 - computed_proby1gty2) < ALMOST_ZERO
    end end
    println("End test_probresults_from_lambdas_double_poisson_equivalence")
end

function test_probresults_from_means_double_poisson_equivalence()::Nothing
    # prob(y1 < y2, mean1, mean2, corr12 = 0.0) == sum(y1 in 0:Inf, y2 in 0:Inf, where y1 < y2: poisson.pdf(y1, mean1)*poisson.pdf(y2, mean2))
    # prob(y1 > y2, mean1, mean2, corr12 = 0.0) == sum(y1 in 0:Inf, y2 in 0:Inf, where y1 > y2: : poisson.pdf(y1, mean1)*poisson.pdf(y2, mean2))
    println("Start test_probresults_from_means_double_poisson_equivalence")
    for mean1 in LAMBDA_RANGE
        for mean2 in LAMBDA_RANGE
            proby1lty2 = 0.0
            proby1eqy2 = 0.0
            proby1gty2 = 0.0
            probtotal = 0.0
            prev_probtotal = -1.0
            converged = false
            y1 = 0
            while !converged  # for y1 in range(0..) while not converged
                proby1eqy2 += pdf(Poisson(mean1), y1)*pdf(Poisson(mean2), y1)
                for y2 in 0:1:(y1 - 1)
                    proby1lty2 += pdf(Poisson(mean1), y2)*pdf(Poisson(mean2), y1)
                    proby1gty2 += pdf(Poisson(mean1), y1)*pdf(Poisson(mean2), y2)
                end
                probtotal = proby1lty2 + proby1eqy2 + proby1gty2
                converged = (probtotal == prev_probtotal)
                prev_probtotal = probtotal
                y1 += 1
            end
            expected_proby1lty2, expected_proby1gty2 = proby1lty2/probtotal, proby1gty2/probtotal
            computed_proby1lty2, computed_proby1gty2 = probresults_from_means(mean1, mean2, 0.0)
            @test abs(expected_proby1lty2 - computed_proby1lty2) < ALMOST_ZERO
            @test abs(expected_proby1gty2 - computed_proby1gty2) < ALMOST_ZERO
    end end
    println("End test_probresults_from_means_double_poisson_equivalence")
end

# ==============================================================================
# test_lambdas_from_probresults_*()
# ==============================================================================

LAMBDA_TOLERANCE = 1.0e-08

function test_lambdas_from_probresults_basic()::Nothing
    println("Start test_lambdas_from_probresults_basic")
    # probs equal, zero
    @test lambdas_from_probresults(0.0, 0.0, 0.0, LAMBDA_TOLERANCE) == (0.0, 0.0)
    @test lambdas_from_probresults(0.0, 0.0, 1.0, LAMBDA_TOLERANCE) == (0.0, 0.0)

    # probs equal, non-zero
    lambda03_x0, lambdaequal = lambdas_from_probresults(0.3, 0.3, 0.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda03_x0 == lambdaequal < Inf
    lambda03_x1, lambdaequal = lambdas_from_probresults(0.3, 0.3, 1.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda03_x1 == lambdaequal < Inf
    @test abs(lambda03_x0 - lambda03_x1) < LAMBDA_TOLERANCE
    lambda04_x0, lambdaequal = lambdas_from_probresults(0.4, 0.4, 0.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda03_x0 < lambda04_x0 == lambdaequal < Inf
    lambda04_x1, lambdaequal = lambdas_from_probresults(0.4, 0.4, 1.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda03_x1 < lambda04_x1 == lambdaequal < Inf
    @test abs(lambda04_x0 - lambda04_x1) < LAMBDA_TOLERANCE

    # one prob zero, other prob non-zero
    zero, lambda_x0 = lambdas_from_probresults(0.5, 0.0, 0.0, LAMBDA_TOLERANCE)
    @test 0.0 == zero < lambda_x0 < Inf
    @test lambdas_from_probresults(0.0, 0.5, 0.0, LAMBDA_TOLERANCE) == (lambda_x0, 0.0)
    zero, lambda_x1 = lambdas_from_probresults(0.5, 0.0, 1.0, LAMBDA_TOLERANCE)
    @test 0.0 == zero < lambda_x1 < Inf
    @test abs(lambda_x0 - lambda_x1) < LAMBDA_TOLERANCE
    @test lambdas_from_probresults(0.0, 0.5, 1.0, LAMBDA_TOLERANCE) == (lambda_x1, 0.0)

    # probs non-equal, non-zero
    lambda02_x0, lambda04_x0 = lambdas_from_probresults(0.2, 0.4, 0.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda04_x0 < lambda02_x0 < Inf
    @test lambdas_from_probresults(0.4, 0.2, 0.0, LAMBDA_TOLERANCE) == (lambda04_x0, lambda02_x0)
    lambda02_x1, lambda04_x1 = lambdas_from_probresults(0.2, 0.4, 1.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda04_x1 < lambda02_x1 < Inf
    @test abs(lambda02_x0 - lambda02_x1) < LAMBDA_TOLERANCE
    @test abs(lambda04_x0 - lambda04_x1) < LAMBDA_TOLERANCE
    @test lambdas_from_probresults(0.4, 0.2, 1.0, LAMBDA_TOLERANCE) == (lambda04_x1, lambda02_x1)

    lambda01_x0, lambda06_x0 = lambdas_from_probresults(0.1, 0.6, 0.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda06_x0 < lambda04_x0 < lambda02_x0 < lambda01_x0 < Inf
    @test lambdas_from_probresults(0.6, 0.1, 0.0, LAMBDA_TOLERANCE) == (lambda06_x0, lambda01_x0)
    lambda01_x1, lambda06_x1 = lambdas_from_probresults(0.1, 0.6, 1.0, LAMBDA_TOLERANCE)
    @test 0.0 < lambda06_x1 < lambda04_x1 < lambda02_x1 < lambda01_x1 < Inf
    @test abs(lambda01_x0 - lambda01_x1) < LAMBDA_TOLERANCE
    @test abs(lambda06_x0 - lambda06_x1) < LAMBDA_TOLERANCE
    @test lambdas_from_probresults(0.6, 0.1, 1.0, LAMBDA_TOLERANCE) == (lambda06_x1, lambda01_x1)
    println("End test_lambdas_from_probresults_basic")
end

function test_lambdas_from_probresults_axioms()::Nothing
    println("Start test_lambdas_from_probresults_axioms")
    for proby1lty2 in PROB_RANGE
        for proby1gty2 in PROB_RANGE
            if proby1lty2 + proby1gty2 < 1.0
                lambda1, lambda2 = lambdas_from_probresults(proby1lty2, proby1gty2, 0.0, LAMBDA_TOLERANCE)
                @test 0.0 <= lambda1 && 0.0 <= lambda2
                @test (lambda1 == 0.0 && lambda2 == 0.0) == (proby1lty2 == 0.0 && proby1gty2 == 0.0)
                @test 0.0 < lambda1 || proby1gty2 == 0
                @test 0.0 < lambda2 || proby1lty2 == 0
                @test (abs(lambda1 - lambda2) < LAMBDA_TOLERANCE) == (proby1lty2 == proby1gty2)
                for lambdax in LAMBDAX_RANGE[2:length(LAMBDAX_RANGE)]
                    lambda1x, lambda2x = lambdas_from_probresults(proby1lty2, proby1gty2, lambdax, LAMBDA_TOLERANCE)
                    @test abs(lambda1 - lambda1x) < 16*LAMBDA_TOLERANCE && abs(lambda2 - lambda2x) < 16*LAMBDA_TOLERANCE
                end
    end end end
    println("End test_lambdas_from_probresults_axioms")
end

function test_lambdas_from_probresults_symmetry()::Nothing
    println("Start test_lambdas_from_probresults_symmetry")
    # Don't need to iterate over lambdax because lambdax doesn't affect probresults.
    # lambdas_from_probresults(cdf(lambdas, lambdax), lambdax) == lambdas
    for lambda1 in LAMBDA_RANGE
        for lambda2 in LAMBDA_RANGE
            proby1lty2, proby1gty2 = probresults_from_lambdas(lambda1, lambda2, 0.0)
            lambda1result, lambda2result = lambdas_from_probresults(proby1lty2, proby1gty2, 0.0, LAMBDA_TOLERANCE)
            @test abs(lambda1result - lambda1) < 16*LAMBDA_TOLERANCE && abs(lambda2result - lambda2) < 16*LAMBDA_TOLERANCE
    end end
    # probresults_from_lambdas(lambdas_from_probresults(probs)) == probs
    for proby1lty2 in PROB_RANGE
        for proby1gty2 in PROB_RANGE
            if proby1lty2 + proby1gty2 < 1.0
                lambda1, lambda2 = lambdas_from_probresults(proby1lty2, proby1gty2, 0.0, LAMBDA_TOLERANCE)
                proby1lty2result, proby1gty2result = probresults_from_lambdas(lambda1, lambda2, 0.0)
                @test abs(proby1lty2result - proby1lty2) < 16*LAMBDA_TOLERANCE && abs(proby1gty2result - proby1gty2) < 16*LAMBDA_TOLERANCE
    end end end
    println("End test_lambdas_from_probresults_symmetry")
end

# ==============================================================================
# test_means_from_probresults_*()
# ==============================================================================

MEAN_TOLERANCE = 1.0e-08

function test_means_from_probresults_basic()::Nothing
    println("Start test_means_from_probresults_basic")
    # probs equal, zero
    @test means_from_probresults(0.0, 0.0, 0.0, MEAN_TOLERANCE) == (0.0, 0.0)
    @test means_from_probresults(0.0, 0.0, 0.5, MEAN_TOLERANCE) == (0.0, 0.0)
    @test means_from_probresults(0.0, 0.0, 1.0, MEAN_TOLERANCE) == (0.0, 0.0)

    # probs equal, non-zero
    mean03_c00, meanequal = means_from_probresults(0.3, 0.3, 0.0, MEAN_TOLERANCE)
    @test 0.0 < mean03_c00 == meanequal < Inf
    mean03_c05, meanequal = means_from_probresults(0.3, 0.3, 0.5, MEAN_TOLERANCE)
    @test 0.0 < mean03_c05 == meanequal < Inf
    @test mean03_c00 < mean03_c05
    mean04_c00, meanequal = means_from_probresults(0.4, 0.4, 0.0, MEAN_TOLERANCE)
    @test 0.0 < mean03_c00 < mean04_c00 == meanequal < Inf
    mean04_c05, meanequal = means_from_probresults(0.4, 0.4, 0.5, MEAN_TOLERANCE)
    @test 0.0 < mean03_c05 < mean04_c05 == meanequal < Inf
    @test mean04_c00 < mean04_c05

    # one prob zero, other prob non-zero
    mean1_c00, mean2_c00 = means_from_probresults(0.5, 0.0, 0.0, MEAN_TOLERANCE)
    @test 0.0 == mean1_c00 < mean2_c00 < Inf
    @test means_from_probresults(0.0, 0.5, 0.0, MEAN_TOLERANCE) == (mean2_c00, mean1_c00)

    # probs non-equal, non-zero
    mean02_c00, mean04_c00 = means_from_probresults(0.2, 0.4, 0.0, MEAN_TOLERANCE)
    @test 0.0 < mean04_c00 < mean02_c00 < Inf
    @test means_from_probresults(0.4, 0.2, 0.0, MEAN_TOLERANCE) == (mean04_c00, mean02_c00)
    mean02_c05, mean04_c05 = means_from_probresults(0.2, 0.4, 0.5, MEAN_TOLERANCE)
    @test 0.0 < mean04_c05 < mean02_c05 < Inf
    @test means_from_probresults(0.4, 0.2, 0.5, MEAN_TOLERANCE) == (mean04_c05, mean02_c05)
    @test mean02_c00 < mean02_c05
    @test mean04_c00 < mean04_c05

    mean01_c00, mean06_c00 = means_from_probresults(0.1, 0.6, 0.0, MEAN_TOLERANCE)
    @test 0.0 < mean06_c00 < mean04_c00 < mean02_c00 < mean01_c00 < Inf
    @test means_from_probresults(0.6, 0.1, 0.0, MEAN_TOLERANCE) == (mean06_c00, mean01_c00)
    mean01_c05, mean06_c05 = means_from_probresults(0.1, 0.6, 0.5, MEAN_TOLERANCE)
    @test 0.0 < mean06_c05 < mean04_c05 < mean02_c05 < mean01_c05 < Inf
    @test means_from_probresults(0.6, 0.1, 0.5, MEAN_TOLERANCE) == (mean06_c05, mean01_c05)
    @test mean01_c00 < mean01_c05
    @test mean06_c00 < mean06_c05
    println("End test_means_from_probresults_basic")
end

function test_means_from_probresults_axioms()::Nothing
    println("Start test_means_from_probresults_axioms")
    for corr12 in FULL_CORR_RANGE
        @test means_from_probresults(0.0, 0.0, corr12, MEAN_TOLERANCE) == (0.0, 0.0)
    end
    for proby1lty2 in PROB_RANGE
        for proby1gty2 in PROB_RANGE
            if proby1lty2 + proby1gty2 < 1.0
                for corr12 in FULL_CORR_RANGE[1:(length(FULL_CORR_RANGE) - 1)]
                    mean1, mean2 = means_from_probresults(proby1lty2, proby1gty2, corr12, MEAN_TOLERANCE)
                    @test 0.0 <= mean1 < Inf && 0.0 <= mean2 < Inf
                    @test (mean1 == mean2 == 0.0) == (proby1lty2 == proby1gty2 == 0.0)
                    @test meansvalid(mean1, mean2, corr12)
                    @test 0.0 < mean1 || proby1gty2 == 0.0
                    @test 0.0 < mean2 || proby1lty2 == 0.0
                    @test !(proby1lty2 == proby1gty2) || abs(mean1 - mean2) < MEAN_TOLERANCE
    end end end end
    println("End test_means_from_probresults_axioms")
end

function test_means_from_probresults_symmetry()::Nothing
    println("Start test_means_from_probresults_symmetry")
    # means_from_probresults(probresults_from_means(means)) == means
    for mean1 in MEAN_RANGE
        for mean2 in MEAN_RANGE
            if 0.0 < mean1 || 0.0 < mean2
                for corr12 in MEANS_CORR_RANGE(mean1, mean2)
                    proby1lty2, proby1gty2 = probresults_from_means(mean1, mean2, corr12)
                    if 0.0 < proby1lty2 || 0.0 < proby1gty2
                        mean1result, mean2result = means_from_probresults(proby1lty2, proby1gty2, corr12, MEAN_TOLERANCE)
                        @test abs(mean1result - mean1) < 16*MEAN_TOLERANCE && abs(mean2result - mean2) < 16*MEAN_TOLERANCE
    end end end end end
    # probresults_from_means(means_from_probresults(probs)) == probs
    for proby1lty2 in PROB_RANGE
        for proby1gty2 in PROB_RANGE
            if proby1lty2 + proby1gty2 < 1.0
                for corr12 in FULL_CORR_RANGE[1:(length(FULL_CORR_RANGE) - 1)]
                    mean1, mean2 = means_from_probresults(proby1lty2, proby1gty2, corr12, MEAN_TOLERANCE)
                    proby1lty2result, proby1gty2result = probresults_from_means(mean1, mean2, corr12)
                    @test abs(proby1lty2result - proby1lty2) < 16*MEAN_TOLERANCE && abs(proby1gty2result - proby1gty2) < 16*MEAN_TOLERANCE
    end end end end
    println("End test_means_from_probresults_symmetry")
end

# ==============================================================================
# test_prob*_from_prob*andlambda*_*()
# ==============================================================================

PROB_TOLERANCE = 1.0e-04

function test_proby1lty2_from_proby1gty2andlambda1_basic()
    println("Start test_proby1lty2_from_proby1gty2andlambda1_basic")
    # zero prob of win1, non-zero lambda1 => infinite lambda2, sure prob of win2
    @test proby1lty2_from_proby1gty2andlambda1(0.0, 1.0, 0.0, PROB_TOLERANCE) == 1.0
    @test proby1lty2_from_proby1gty2andlambda1(0.0, 1.0, 1.0, PROB_TOLERANCE) == 1.0
    # half prob of win1 => less than half prob of win2
    @test proby1lty2_from_proby1gty2andlambda1(0.5, 1.0, 0.0, PROB_TOLERANCE) < 0.5
    @test proby1lty2_from_proby1gty2andlambda1(0.5, 1.0, 1.0, PROB_TOLERANCE) < 0.5
    # high prob of win1, low lambda1 => zero lambda2, no prob of win2
    @test isnan(proby1lty2_from_proby1gty2andlambda1(0.9, 1.0, 0.0, PROB_TOLERANCE))
    @test isnan(proby1lty2_from_proby1gty2andlambda1(0.9, 1.0, 1.0, PROB_TOLERANCE))
    println("End test_proby1lty2_from_proby1gty2andlambda1_basic")
end

function test_proby1lty2_from_proby1gty2andlambda1_axioms()
    println("Start test_proby1lty2_from_proby1gty2andlambda1_axioms")
    for lambda1 in LAMBDA_RANGE[2:length(LAMBDA_RANGE)]
        priorproby1lty2 = Inf
        for proby1gty2 in PROB_RANGE[1:(length(PROB_RANGE) - 1)]
            println("proby1lty2_from_proby1gty2andlambda1(proby1gty2 = $(proby1gty2), lambda1 = $(lambda1), lambdax = 0.0, PROB_TOLERANCE = $(PROB_TOLERANCE))")
            proby1lty2 = proby1lty2_from_proby1gty2andlambda1(proby1gty2, lambda1, 0.0, PROB_TOLERANCE)
            if !isnan(proby1lty2)
                @test !isnan(priorproby1lty2)
                @test proby1lty2 < priorproby1lty2
                if proby1lty2 + proby1lty2 < 0.99
                    println("lambdas_from_probresults(proby1lty2 = $(proby1lty2), proby1gty2 = $(proby1gty2), lambdax= $(0.0), PROB_TOLERANCE/4)")
                    lambda1result, lambda2result = lambdas_from_probresults(proby1lty2, proby1gty2, 0.0, PROB_TOLERANCE/4)
                    @test abs(lambda1 - lambda1result) < PROB_TOLERANCE
                end
            end
            for lambdax in LAMBDAX_RANGE[2:length(LAMBDAX_RANGE)]
                println("proby1lty2_from_proby1gty2andlambda1(proby1gty2 = $(proby1gty2), lambda1 = $(lambda1), lambdax = $(lambdax), PROB_TOLERANCE = $(PROB_TOLERANCE))")
                proby1lty2x = proby1lty2_from_proby1gty2andlambda1(proby1gty2, lambda1, lambdax, PROB_TOLERANCE)
                @test isnan(proby1lty2) && isnan(proby1lty2x) || abs(proby1lty2 - proby1lty2x) < PROB_TOLERANCE
            end
            priorproby1lty2 = proby1lty2
    end end
    println("End test_proby1lty2_from_proby1gty2andlambda1_axioms")
end

function test_proby1gty2_from_proby1lty2andlambda1_basic()
    println("Start test_proby1gty2_from_proby1lty2andlambda1_basic")
    # zero prob of win2, non-zero lambda1 => non-zero, non-infinite prob of win1
    @test 0.0 < proby1gty2_from_proby1lty2andlambda1(0.0, 1.0, 0.0, PROB_TOLERANCE) < 1.0
    @test 0.0 < proby1gty2_from_proby1lty2andlambda1(0.0, 1.0, 1.0, PROB_TOLERANCE) < 1.0
    # half prob of win2 => less than half prob of win1
    @test proby1gty2_from_proby1lty2andlambda1(0.5, 1.0, 0.0, PROB_TOLERANCE) < 0.5
    @test proby1gty2_from_proby1lty2andlambda1(0.5, 1.0, 1.0, PROB_TOLERANCE) < 0.5
    # high prob of win1 => non-zero prob of win2
    @test 0.0 < proby1gty2_from_proby1lty2andlambda1(0.9, 1.0, 0.0, PROB_TOLERANCE) < 0.1
    @test 0.0 < proby1gty2_from_proby1lty2andlambda1(0.9, 1.0, 1.0, PROB_TOLERANCE) < 0.1
    println("End test_proby1gty2_from_proby1lty2andlambda1_basic")
end

function test_proby1gty2_from_proby1lty2andlambda1_axioms()
    println("Start test_proby1gty2_from_proby1lty2andlambda1_axioms")
    for lambda1 in LAMBDA_RANGE[2:length(LAMBDA_RANGE)]
        priorproby1gty2 = Inf
        for proby1lty2 in PROB_RANGE
            println("proby1gty2_from_proby1lty2andlambda1(proby1lty2 = $(proby1lty2), lambda1 = $(lambda1), lambdax = 0.0, PROB_TOLERANCE = $(PROB_TOLERANCE))")
            proby1gty2 = proby1gty2_from_proby1lty2andlambda1(proby1lty2, lambda1, 0.0, PROB_TOLERANCE)
            @test proby1gty2 < priorproby1gty2
            if proby1lty2 + proby1lty2 < 0.99
                lambda1result, lambda2result = lambdas_from_probresults(proby1lty2, proby1gty2, 0.0, PROB_TOLERANCE/4)
                @test abs(lambda1 - lambda1result) < PROB_TOLERANCE
            end
            for lambdax in LAMBDAX_RANGE[2:length(LAMBDAX_RANGE)]
                println("proby1gty2_from_proby1lty2andlambda1(proby1lty2 = $(proby1lty2), lambda1 = $(lambda1), lambdax = $(lambdax), PROB_TOLERANCE = $(PROB_TOLERANCE))")
                proby1gty2x = proby1gty2_from_proby1lty2andlambda1(proby1lty2, lambda1, lambdax, PROB_TOLERANCE)
                @test abs(proby1gty2 - proby1gty2x) < PROB_TOLERANCE
            end
            priorproby1gty2 = proby1gty2
    end end
    println("End test_proby1gty2_from_proby1lty2andlambda1_axioms")
end

# ==============================================================================
# test_prob*_from_prob*andmean*_*()
# ==============================================================================

PROB_TOLERANCE = 1.0e-04

function test_proby1lty2_from_proby1gty2andmean1_basic()
    println("Start test_proby1lty2_from_proby1gty2andmean1_basic")
    # zero prob of win1
    proby1lty2_p00_c00 = proby1lty2_from_proby1gty2andmean1(0.0, 1.0, 0.0, PROB_TOLERANCE)
    proby1lty2_p00_c25 = proby1lty2_from_proby1gty2andmean1(0.0, 1.0, 0.25, PROB_TOLERANCE)
    proby1lty2_p00_c50 = proby1lty2_from_proby1gty2andmean1(0.0, 1.0, 0.5, PROB_TOLERANCE)
    @test 0.0 < proby1lty2_p00_c50 < proby1lty2_p00_c25 < proby1lty2_p00_c00 == 1.0
    # 0.25 prob of win1
    proby1lty2_p25_c00 = proby1lty2_from_proby1gty2andmean1(0.25, 1.0, 0.0, PROB_TOLERANCE)
    proby1lty2_p25_c25 = proby1lty2_from_proby1gty2andmean1(0.25, 1.0, 0.25, PROB_TOLERANCE)
    proby1lty2_p25_c50 = proby1lty2_from_proby1gty2andmean1(0.25, 1.0, 0.5, PROB_TOLERANCE)
    @test 0.0 < proby1lty2_p25_c50 < proby1lty2_p25_c25 < proby1lty2_p25_c00
    # 0.5 prob of win1
    proby1lty2_p50_c00 = proby1lty2_from_proby1gty2andmean1(0.5, 1.0, 0.0, PROB_TOLERANCE)
    proby1lty2_p50_c25 = proby1lty2_from_proby1gty2andmean1(0.5, 1.0, 0.25, PROB_TOLERANCE)
    proby1lty2_p50_c50 = proby1lty2_from_proby1gty2andmean1(0.5, 1.0, 0.5, PROB_TOLERANCE)
    @test 0.0 < proby1lty2_p50_c50 < proby1lty2_p50_c25 < proby1lty2_p50_c00
    @test 0.0 < proby1lty2_p50_c00 < proby1lty2_p25_c00 < proby1lty2_p00_c00
    @test 0.0 < proby1lty2_p50_c25 < proby1lty2_p25_c25 < proby1lty2_p00_c25
    @test 0.0 < proby1lty2_p50_c50 < proby1lty2_p25_c50 < proby1lty2_p00_c50
    # high prob of win1, low mean1 => no prob of win2
    @test isnan(proby1lty2_from_proby1gty2andmean1(0.9, 1.0, 0.0, PROB_TOLERANCE))
    @test isnan(proby1lty2_from_proby1gty2andmean1(0.9, 1.0, 0.5, PROB_TOLERANCE))
    @test isnan(proby1lty2_from_proby1gty2andmean1(0.9, 1.0, 0.9, PROB_TOLERANCE))
    # corr12 == 1.0
    @test proby1lty2_from_proby1gty2andmean1(0.0, 0.0, 1.0, PROB_TOLERANCE) == 0.0
    @test proby1lty2_from_proby1gty2andmean1(0.0, 1.0, 1.0, PROB_TOLERANCE) == 0.0
    @test isnan(proby1lty2_from_proby1gty2andmean1(0.5, 0.0, 1.0, PROB_TOLERANCE))
    @test isnan(proby1lty2_from_proby1gty2andmean1(0.5, 1.0, 1.0, PROB_TOLERANCE))
    println("End test_proby1lty2_from_proby1gty2andmean1_basic")
end

function test_proby1lty2_from_proby1gty2andmean1_axioms()
    println("Start test_proby1lty2_from_proby1gty2andmean1_axioms")
    for mean1 in MEAN_RANGE[2:length(MEAN_RANGE)]
        for corr12 in FULL_CORR_RANGE[1:(length(FULL_CORR_RANGE) - 1)]
            priorproby1lty2 = Inf
            for proby1gty2 in PROB_RANGE[1:(length(PROB_RANGE) - 1)]
                println("proby1lty2_from_proby1gty2andmean1(proby1gty2 = $(proby1gty2), mean1 = $(mean1), corr12 = $(corr12), PROB_TOLERANCE = $(PROB_TOLERANCE))")
                proby1lty2 = proby1lty2_from_proby1gty2andmean1(proby1gty2, mean1, corr12, PROB_TOLERANCE)
                if !isnan(proby1lty2)
                    @test !isnan(priorproby1lty2)
                    @test proby1lty2 < priorproby1lty2
                    if proby1lty2 + proby1gty2 < 0.99
                        println("means_from_probresults(proby1lty2 = $(proby1lty2), proby1gty2 = $(proby1gty2), corr12= $(corr12), PROB_TOLERANCE/4)")
                        mean1result, mean2result = means_from_probresults(proby1lty2, proby1gty2, corr12, PROB_TOLERANCE/4)
                        @test abs(mean1 - mean1result) < PROB_TOLERANCE
                    end
                priorproby1lty2 = proby1lty2
    end end end end
    println("End test_proby1lty2_from_proby1gty2andmean1_axioms")
end

function test_proby1gty2_from_proby1lty2andmean1_basic()
    println("Start test_proby1gty2_from_proby1lty2andmean1_basic")
    # zero prob of win2
    proby1gty2_p00_c00 = proby1gty2_from_proby1lty2andmean1(0.0, 1.0, 0.0, PROB_TOLERANCE)
    proby1gty2_p00_c25 = proby1gty2_from_proby1lty2andmean1(0.0, 1.0, 0.25, PROB_TOLERANCE)
    proby1gty2_p00_c50 = proby1gty2_from_proby1lty2andmean1(0.0, 1.0, 0.5, PROB_TOLERANCE)
    @test 0.0 < proby1gty2_p00_c50 < proby1gty2_p00_c25 < proby1gty2_p00_c00 < 1.0
    # 0.25 prob of win2
    proby1gty2_p25_c00 = proby1gty2_from_proby1lty2andmean1(0.25, 1.0, 0.0, PROB_TOLERANCE)
    proby1gty2_p25_c25 = proby1gty2_from_proby1lty2andmean1(0.25, 1.0, 0.25, PROB_TOLERANCE)
    proby1gty2_p25_c50 = proby1gty2_from_proby1lty2andmean1(0.25, 1.0, 0.5, PROB_TOLERANCE)
    @test 0.0 < proby1gty2_p25_c50 < proby1gty2_p25_c25 < proby1gty2_p25_c00
    # 0.5 prob of win2
    proby1gty2_p50_c00 = proby1gty2_from_proby1lty2andmean1(0.5, 1.0, 0.0, PROB_TOLERANCE)
    proby1gty2_p50_c25 = proby1gty2_from_proby1lty2andmean1(0.5, 1.0, 0.25, PROB_TOLERANCE)
    proby1gty2_p50_c50 = proby1gty2_from_proby1lty2andmean1(0.5, 1.0, 0.5, PROB_TOLERANCE)
    @test 0.0 < proby1gty2_p50_c50 < proby1gty2_p50_c25 < proby1gty2_p50_c00
    @test 0.0 < proby1gty2_p50_c00 < proby1gty2_p25_c00 < proby1gty2_p00_c00
    @test 0.0 < proby1gty2_p50_c25 < proby1gty2_p25_c25 < proby1gty2_p00_c25
    @test 0.0 < proby1gty2_p50_c50 < proby1gty2_p25_c50 < proby1gty2_p00_c50
    # high prob of win2, low mean1 => low changing to no prob of win1
    proby1gty2_p90_c00 = proby1gty2_from_proby1lty2andmean1(0.9, 1.0, 0.0, PROB_TOLERANCE)
    @test !isnan(proby1gty2_p90_c00)
    proby1gty2_p90_c50 = proby1gty2_from_proby1lty2andmean1(0.9, 1.0, 0.5, PROB_TOLERANCE)
    @test !isnan(proby1gty2_p90_c50)
    @test proby1gty2_p90_c50 < proby1gty2_p90_c00
    proby1gty2_p90_c90 = proby1gty2_from_proby1lty2andmean1(0.9, 1.0, 0.9, PROB_TOLERANCE)
    @test isnan(proby1gty2_p90_c90)
    # corr12 == 1.0
    @test proby1gty2_from_proby1lty2andmean1(0.0, 0.0, 1.0, PROB_TOLERANCE) == 0.0
    @test proby1gty2_from_proby1lty2andmean1(0.0, 1.0, 1.0, PROB_TOLERANCE) == 0.0
    @test isnan(proby1gty2_from_proby1lty2andmean1(0.5, 0.0, 1.0, PROB_TOLERANCE))
    @test isnan(proby1gty2_from_proby1lty2andmean1(0.5, 1.0, 1.0, PROB_TOLERANCE))
    println("End test_proby1gty2_from_proby1lty2andmean1_basic")
end

function test_proby1gty2_from_proby1lty2andmean1_axioms()
    println("Start test_proby1gty2_from_proby1lty2andmean1_axioms")
    for mean1 in MEAN_RANGE[2:length(MEAN_RANGE)]
        for corr12 in FULL_CORR_RANGE[1:(length(FULL_CORR_RANGE) - 1)]
            priorproby1gty2 = Inf
            for proby1lty2 in PROB_RANGE[1:(length(PROB_RANGE) - 1)]
                println("proby1gty2_from_proby1lty2andmean1(proby1lty2 = $(proby1lty2), mean1 = $(mean1), corr12 = $(corr12), PROB_TOLERANCE = $(PROB_TOLERANCE))")
                proby1gty2 = proby1gty2_from_proby1lty2andmean1(proby1lty2, mean1, corr12, PROB_TOLERANCE)
                if !isnan(proby1gty2)
                    @test !isnan(priorproby1gty2)
                    @test proby1gty2 < priorproby1gty2
                    if proby1lty2 + proby1gty2 <= 0.99
                        println("means_from_probresults(proby1lty2 = $(proby1lty2), proby1gty2 = $(proby1gty2), corr12= $(corr12), PROB_TOLERANCE/4)")
                        mean1result, mean2result = means_from_probresults(proby1lty2, proby1gty2, corr12, PROB_TOLERANCE/4)
                        @test abs(mean1 - mean1result) < PROB_TOLERANCE
                    end
                priorproby1gty2 = proby1gty2
    end end end end
    println("End test_proby1gty2_from_proby1lty2andmean1_axioms")
end

# ==============================================================================
# Main program
# ==============================================================================

function test_probscore_from_lambdas()::Nothing
    println("Start test_probscore_from_lambdas")
    test_probscore_from_lambdas_basic()
    test_probscore_from_lambdas_sums()
    test_probscore_from_lambdas_axioms()
    test_probscore_from_lambdas_symmetry()
    test_probscore_from_lambdas_double_poisson_equivalence()
    test_probscore_from_lambdas_marginal_poisson()
    test_probscore_from_lambdas_cummulative_stats()
    println("End test_probscore_from_lambdas")
end

function test_means_arithmetic()::Nothing
    println("Start test_means_arithmetic")
    test_meansvalid_basic()
    test_minsmallermean_basic()
    test_maxlargermean_basic()
    test_maxcorr12_basic()
    test_means_axioms()
    test_means_range()
    println("End test_means_arithmetic")
end

function test_probscore_from_means()::Nothing
    println("Start test_probscore_from_lambdas")
    test_probscore_from_means_basic()
    test_probscore_from_means_sums()
    test_probscore_from_means_axioms()
    test_probscore_from_means_symmetry()
    test_probscore_from_means_double_poisson_equivalence()
    test_probscore_from_means_marginal_poisson()
    test_probscore_from_means_cummulative_stats()
    println("End test_probscore_from_lambdas")
end

function test_probresults_from_lambdas()::Nothing
    println("Start test_probresults_from_lambdas")
    test_probresults_from_lambdas_basic()
    test_probresults_from_lambdas_axioms()
    test_probresults_from_lambdas_relations()
    test_probresults_from_lambdas_double_poisson_equivalence()
    println("End test_probresults_from_lambdas")
end

function test_probresults_from_means()::Nothing
    println("Start test_probresults_from_means")
    test_probresults_from_means_basic()
    test_probresults_from_means_axioms()
    test_probresults_from_means_equal()
    test_probresults_from_means_symmetry()
    test_probresults_from_means_double_poisson_equivalence()
    println("End test_probresults_from_means")
end

function test_lambdas_from_probresults()::Nothing
    println("Start test_lambdas_from_probresults")
    test_lambdas_from_probresults_basic()
    test_lambdas_from_probresults_axioms()
    test_lambdas_from_probresults_symmetry()
    println("End test_lambdas_from_probresults")
end

function test_means_from_probresults()::Nothing
    println("Start test_means_from_probresults")
    test_means_from_probresults_basic()
    test_means_from_probresults_axioms()
    test_means_from_probresults_symmetry()
    println("End test_means_from_probresults")
end

function test_prob_from_probandlambda()::Nothing
    println("Start test_prob_from_probandlambda")
    test_proby1lty2_from_proby1gty2andlambda1_basic()
    test_proby1lty2_from_proby1gty2andlambda1_axioms()
    test_proby1gty2_from_proby1lty2andlambda1_basic()
    test_proby1gty2_from_proby1lty2andlambda1_axioms()
    println("End test_prob_from_probandlambda")
end

function test_prob_from_probandmean()::Nothing
    println("Start test_prob_from_probandmean")
    test_proby1lty2_from_proby1gty2andmean1_basic()
    test_proby1lty2_from_proby1gty2andmean1_axioms()
    test_proby1gty2_from_proby1lty2andmean1_basic()
    test_proby1gty2_from_proby1lty2andmean1_axioms()
    println("End test_prob_from_probandmean")
end

function main()::Nothing
    println("Start main")
    test_probscore_from_lambdas()
    test_means_arithmetic()
    test_probscore_from_means()
    test_probresults_from_lambdas()
    test_probresults_from_means()
    test_lambdas_from_probresults()
    test_means_from_probresults()
    test_prob_from_probandlambda()
    test_prob_from_probandmean()
    println("End main")
end

main()

# ==============================================================================
end
