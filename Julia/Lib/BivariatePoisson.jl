module BivariatePoisson
# ==============================================================================
# BivariatePoisson
#
# Probability functions for two random variables, Y1 and Y2,
# distributed with a joint bivariate Poisson distribution.
#
# The canonical definition is in terms of the (lambda1, lambda2, lambda3)
# parameterization. However, for our purpose, the useful definition is in terms
# of the (mean1, mean2, corr12) paramaterization.
#
# The probability distribution functions probscore_from_lambdas(),
# probscore_from_means(), probresults_from_lambdas, and probresults_from_means()
# are computed directly.  The inverse functions lambdas_from_probresults() and
# means_from_probresults() are computed numerically using NLsolve.
# ==============================================================================

export clear_caches!
export probscore_from_lambdas, probscore_from_means
export meansvalid, maxcorr12, minsmallermean, maxlargermean
export probresults_from_lambdas, probresults_from_means
export lambdas_from_probresults, means_from_probresults
export proby1lty2_from_proby1gty2andlambda1, proby1gty2_from_proby1lty2andlambda1
export proby1lty2_from_proby1gty2andmean1, proby1gty2_from_proby1lty2andmean1

using NLsolve  # See: https://libraries.io/julia/NLsolve

# ==============================================================================
# Memoization caches
# TODO: Replace with https://github.com/marius311/Memoization.jl
# ==============================================================================

function clear_caches!()::Nothing
    global probscore_from_lambdas_cache = Dict{Tuple{Int64, Int64, Float64, Float64, Float64}, Float64}()
    global probresults_from_lambdas_cache = Dict{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64}}()
    global lambdas_from_probresults_cache = Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64}}()
    global means_from_probresults_cache = Dict{Tuple{Float64, Float64, Float64, Float64}, Tuple{Float64, Float64}}()
    global proby1lty2_from_proby1gty2andlambda1_cache = Dict{Tuple{Float64, Float64, Float64, Float64}, Float64}()
    global proby1gty2_from_proby1lty2andlambda1_cache = Dict{Tuple{Float64, Float64, Float64, Float64}, Float64}()
    global proby1lty2_from_proby1gty2andmean1_cache = Dict{Tuple{Float64, Float64, Float64, Float64}, Float64}()
    global proby1gty2_from_proby1lty2andmean1_cache = Dict{Tuple{Float64, Float64, Float64, Float64}, Float64}()
    return
end

clear_caches!()

# ==============================================================================
# probscore_from_lambdas(y1, y2; lambda1, lambda2, lambda3) -> Pr(Y1=y1, Y2=y2)
# probscore_from_means(y1, y2; mean1, mean2, corr12) -> Pr(Y1=y1, Y2=y2)
# ==============================================================================

function probscore_from_lambdas(y1::Int64, y2::Int64, lambda1::Float64, lambda2::Float64, lambdax::Float64)::Float64
#
# Pr(Y1=y1, Y2=y2; lambda1, lambda2, lambdax) with a bivariate Poisson distribution parameterized as:
#
#     X1 ~ Poisson(lambda1), X2 ~ Poisson(lambda2), X3 ~ Poisson(lambdax)
#     Y1 = X1 + X3, Y2 = X2 + X3
#
# mean(Y1) = mean(X1) + mean(X3) = lambda1 + lambdax
# mean(Y2) = mean(X2) + mean(X3) = lambda2 + lambdax
# var(Y1) = var(X1) + var(X3) = lambda1 + lambdax
# var(Y2) = var(X2) + var(X3) = lambda2 + lambdax
# covariance(Y1, Y2) = lambdax
# correlation(Y1, Y2) = covariance(Y1, Y2)/sqrt(var(Y1)*var(Y2))
#                     = lambdax/sqrt((lambda1 + lambdax)*(lambda2 + lambdax))
#
# See: https://en.wikipedia.org/wiki/Poisson_distribution#Bivariate_Poisson_distribution
# See: "On bivariate Poisson regression models" by Fatimah E. AlMuhayfith, Abdulhamid A. Alzaid, and Maha A. Omair.
# Journal of King Saud University - Science.  Volume 28, Issue 2, April 2016, Pages 178-189.
# https://www.sciencedirect.com/science/article/pii/S1018364715000798
#
# prob(y1, y2; lambda1, lambda2, lambdax)
#  =  exp(-(lambda1 + lambda2 + lambdax))*((lambda1**y1)/factorial(y1))*((lambda2**y2)/factorial(y2))*
#     sum(r = 0..min(y1, y2) : (y1 C r)*(y2 C r)*factorial(r)*
#                               (lambdax/(lambda1*lambda2))**r)
#  =  exp(-(lambda1 + lambda2 + lambdax))*
#     sum(r = 0..min(y1, y2) : ((y1 C r)/factorial(y1))*((y2 C r)/factorial(y2))*factorial(r)*
#                               (lambdax**r)*(lambda1**(-r))*(lambda2**(-r))*(lambda1**y1)*(lambda2**y2))
#  =  exp(-(lambda1 + lambda2 + lambdax))*
#     sum(r = 0..min(y1, y2) : (1/(factorial(y1 - r)*factorial(r))*(1/(factorial(y2 - r)*factorial(r)))*factorial(r)*
#                               (lambdax**r)*(lambda1**(-r))*(lambda2**(-r))*(lambda1**y1)*(lambda2**y2))
#  =  exp(-(lambda1 + lambda2 + lambdax))*
#     sum(r = 0..min(y1, y2) : (lambdax**r)*(lambda1**(y1 - r))*(lambda2**(y2 - r)/
#                               (factorial(y1 - r)*(factorial(y2 - r)*factorial(r)))
#  =  exp(-(lambda1 + lambda2 + lambdax))*
#     sum(r = 0..min(y1, y2) : (lambdax**r)/factorial(r)*
#                              (lambda1**(y1 - r))/factorial(y1 - r)*
#                              (lambda2**(y2 - r))/factorial(y2 - r))
#
# The transformation avoids divide by zero errors when lambda1 == 0 or lambda2 == 0.
#
    msg = "probscore_from_lambdas(y1 = $(y1), y2 = $(y2), lambda1 = $(lambda1), lambda2 = $(lambda2), lambdax = $(lambdax))"
    MAX_LAMBDA = 745.0  # exp(-746) < eps(0.0)
    @assert 0 <= y1 [msg]
    @assert 0 <= y2 [msg]
    @assert 0.0 <= lambda1 < MAX_LAMBDA [msg]
    @assert 0.0 <= lambda2 < MAX_LAMBDA [msg]
    @assert 0.0 <= lambdax < MAX_LAMBDA [msg]
    if (y1 > y2) || (y1 == y2 && lambda1 > lambda2)  # Ensure numerical symmetry
        probscore = probscore_from_lambdas(y2, y1, lambda2, lambda1, lambdax)
    else
        key = (y1, y2, lambda1, lambda2, lambdax)
        probscore = get(probscore_from_lambdas_cache, key, -1.0)
        if probscore == -1.0
            lambdaxfactor = Vector{Float64}(undef, y1 + 1)  # [0:y1]
            lambda1factor = Vector{Float64}(undef, y1 + 1)  # [0:y1]
            lambda2factor = Vector{Float64}(undef, y1 + 1)  # [(y2 - y1):y2]
            lambdaxfactor[1] = 1.0  # lambdax^0/factorial(0)
            lambda1factor[1] = 1.0  # lambda1^0/factorial(0)
            lambda2factor[1] = prod([lambda2/i for i in 1:(y2 - y1)])  # lambda2^(y2 - y1)/factorial(y2 - y1)
            for r in 1:y1
                lambdaxfactor[r + 1] = lambdaxfactor[r]*lambdax/r
                lambda1factor[r + 1] = lambda1factor[r]*lambda1/r
                lambda2factor[r + 1] = lambda2factor[r]*lambda2/(y2 - y1 + r)
            end
            explambdax = exp(-lambdax)
            explambda1 = exp(-lambda1)
            explambda2 = exp(-lambda2)
            probscore = 0.0
            for r = 0:y1
                probscore += explambdax*lambdaxfactor[r + 1]*explambda1*lambda1factor[y1 - r + 1]*explambda2*lambda2factor[y1 - r + 1]
            end
            global probscore_from_lambdas_cache[key] = probscore
        end
    end
    @assert 0.0 <= probscore <= 1.0
    return probscore
end

function probscore_from_means(y1::Int64, y2::Int64, mean1::Float64, mean2::Float64, corr12::Float64)::Float64
#
# Pr(Y1=y1, Y2=y2; mean1, mean2, corr12) with a bivariate Poisson distribution parameterized as:
#
#     mean(Y1) = mean1, mean(Y2) = mean2, correlation(Y1, Y2) = corr12
#
    msg = "probscore_from_means(y1 =$(y1), y2 = $(y2), mean1 = $(mean1), mean2 = $(mean2), corr12 = $(corr12))"
    @assert 0 <= y1 [msg]
    @assert 0 <= y2 [msg]
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    @assert meansvalid(mean1, mean2, corr12)
    if (y1 > y2) || (y1 == y2 && mean1 > mean2)  # Ensure numerical symmetry
        probscore = probscore_from_means(y2, y1, mean2, mean1, corr12)
    else
        lambda1, lambda2, lambdax = meanstolambdas(mean1, mean2, corr12)
        probscore = probscore_from_lambdas(y1, y2, lambda1, lambda2, lambdax)
    end
    @assert 0.0 <= probscore <= 1.0
    return probscore
end

# ==============================================================================
# lambdas <-> means arithmetic
# ==============================================================================

function meansvalid(mean1::Float64, mean2::Float64, corr12::Float64)::Bool
#
# Is corr12 <= the maximum correlation that is possible between the given pair of bivariate Poisson distribution means.
#
    msg = "meansvalid(mean1 = $(mean1), mean2 = $(mean2), corr12 = $(corr12))"
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= mean2 < Inf [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    if mean1 > mean2  # Ensure numerical symmetry
        return meansvalid(mean2, mean1, corr12)
    else
        return corr12*corr12*max(mean1, mean2) <= min(mean1, mean2) + eps(min(mean1, mean2))
    end
end

function meanstolambdas(mean1::Float64, mean2::Float64, corr12::Float64)::Tuple{Float64, Float64, Float64}
#
# Transform (mean1, mean2, corr12) parameters for probscore_from_means() to
# equivalent (lambda1, lambda2, lambdax) parameters for probscore_from_lambdas().
#
# lambdax = corr12*sqrt(mean1*mean2)
# lambda1 = mean1 - lambdax
# lambda2 = mean2 - lambdax
#
    msg = "meanstolambdas(mean1 = $(mean1), mean2 = $(mean2), corr12 = $(corr12))"
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    @assert meansvalid(mean1, mean2, corr12)
    if mean1 > mean2  # Ensure numerical symmetry
        lambda2, lambda1, lambdax = meanstolambdas(mean2, mean1, corr12)
    else
        lambdax = min(mean1, mean2, corr12*(if (mean1 == mean2) mean1 else sqrt(mean1*mean2) end))
        lambda1 = mean1 - lambdax
        lambda2 = mean2 - lambdax
    end
    @assert 0.0 <= lambda1 < Inf && 0.0 <= lambda2 < Inf && 0.0 <= lambdax < Inf
    return lambda1, lambda2, lambdax
end

function minsmallermean(largermean::Float64, corr12::Float64)::Float64
#
# Minimum smaller mean that is possible given the larger bivariate Poisson distribution mean and the correlation between the two means.
#
# For given largermean and corr12, mininum smallermean => smallermean = (lambdaX + 0) = lambdaX =>
# corr12 = lambdax/sqrt(smallermean*largermean) = smallermean/sqrt(smallermean*largermean) = sqrt(smallermean/largermean) =>
# smallermean = largermean*(corr12**2).
#
    msg = "minsmallermean(largermean = $(largermean), corr12 = $(corr12))"
    @assert 0.0 <= largermean < Inf [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    smallermean = largermean*(corr12^2)
    if !meansvalid(smallermean, largermean, corr12)
        # Adjust smallermean upwards by smallest power of two that resolves floating-point precision
        smallermean_plus = largermean
        eps = 0.5
        while meansvalid(min(smallermean + eps, largermean), largermean, corr12)
            smallermean_plus = min(smallermean + eps, largermean)
            eps = eps/2
        end
        smallermean = smallermean_plus
    end
    @assert 0.0 <= smallermean <= largermean
    return smallermean
end

function maxlargermean(smallermean::Float64, corr12::Float64)::Float64
#
# Maximum larger mean that is possible given the smaller bivariate Poisson distribution mean and the correlation between the two means.
#
# For given smallermean and corr12, maximum largermean => smallermean = (lambdaX + 0) = lambdaX =>
# corr12 = lambdax/sqrt(smallermean*largermean) = smallermean/sqrt(smallermean*largermean) = sqrt(smallermean/largermean) =>
# largermean = smallermean/(corr12**2)
#
    msg = "maxlargermean(smallermean = $(smallermean), corr12 = $(corr12))"
    @assert 0.0 <= smallermean < Inf [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    largermean = if (corr12 == 0.0) Inf else smallermean/(corr12^2) end
    if !(isinf(largermean) || meansvalid(smallermean, largermean, corr12))
        # Adjust largermean downwards by smallest power of two that resolves floating-point precision
        largermean_minus = 0.0
        eps = 0.5
        while meansvalid(smallermean, max(0.0, largermean - eps), corr12)
            largermean_minus = max(0.0, largermean - eps)
            eps = eps/2
        end
        largermean = largermean_minus
    end
    @assert smallermean <= largermean <= Inf
    return largermean
end

function maxcorr12(mean1::Float64, mean2::Float64)::Float64
#
# Maximum correlation that is possible between a given pair of bivariate Poisson distribution means.
#
# For given smallermean and largermean, maximum corr => maximum lambdaX => smallermean = (lambdaX + 0) = lambdaX =>
# corr = lambdax/sqrt(smallermean*largermean) = smallermean/sqrt(smallermean*largermean) = sqrt(smallermean/largermean).
#
    msg = "maxcorr12(mean1 = $(mean1), mean2 = $(mean2))"
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= mean2 < Inf [msg]
    minmean = min(mean1, mean2)
    maxmean = max(mean1, mean2)
    corr12 = if (maxmean == 0.0) 1.0 else min(1.0, sqrt(minmean/maxmean)) end
    if !meansvalid(mean1, mean2, corr12)
        # Adjust corr12 downwards by smallest power of two that resolves floating-point precision
        corr12_minus = 0.0
        eps = 0.5
        while meansvalid(mean1, mean2, max(0.0, corr12 - eps))
            corr12_minus = max(0.0, corr12 - eps)
            eps = eps/2
        end
        corr12 = corr12_minus
    end
    @assert 0.0 <= corr12 <= 1.0
    return corr12
end

# ==============================================================================
# probresults_from_lambdas(lambda1, lambda2, lambda3) -> (Pr(Y1 < Y2), Pr(Y1 > Y2))
# probresults_from_means(mean1, mean2, corr12) -> (Pr(Y1 < Y2), Pr(Y1 > Y2))
# ==============================================================================

function probresults_from_lambdas(lambda1::Float64, lambda2::Float64, lambdax::Float64)::Tuple{Float64, Float64}
#
# (Pr(Y1 < Y2), Pr(Y1 > Y2)) with
# Y1, Y2 ~ probscore_from_lambdas(y1, y2; lambda1, lambda2, lambdax).
#
    msg = "probresults_from_lambdas(lambda1 = $(lambda1), lambda2 = $(lambda2), lambdax = $(lambdax))"
    @assert 0.0 <= lambda1 < Inf [msg]
    @assert 0.0 <= lambda2 < Inf [msg]
    @assert 0.0 <= lambdax < Inf [msg]
    if lambda1 > lambda2  # Ensure numerical symmetry
        proby1gty2, proby1lty2 = probresults_from_lambdas(lambda2, lambda1, lambdax)
    else
        key = (lambda1, lambda2, lambdax)
        proby1lty2, proby1gty2 = get(probresults_from_lambdas_cache, key, (-1.0, -1.0))
        if (proby1lty2, proby1gty2) == (-1.0, -1.0)
            proby1lty2 = 0.0
            proby1eqy2 = 0.0
            proby1gty2 = 0.0
            probtotal = 0.0
            prev_probtotal = -1.0
            converged = false
            y1 = 0
            while !converged  # for y1 in range(0..) while not converged
                probscore = probscore_from_lambdas(y1, y1, lambda1, lambda2, lambdax)
                proby1eqy2 += probscore
                for y2 in 0:1:(y1 - 1)
                    probscore = probscore_from_lambdas(y2, y1, lambda1, lambda2, lambdax)
                    proby1lty2 += probscore
                    probscore = probscore_from_lambdas(y1, y2, lambda1, lambda2, lambdax)
                    proby1gty2 += probscore
                end
                probtotal = proby1lty2 + proby1eqy2 + proby1gty2
                converged = probtotal != 0.0 && (probtotal == prev_probtotal)
                prev_probtotal = probtotal
                y1 += 1
            end
            proby1lty2, proby1gty2 = (proby1lty2, proby1gty2)./probtotal
            global probresults_from_lambdas_cache[key] = (proby1lty2, proby1gty2)
        end
    end
    @assert 0.0 <= proby1lty2 <= 1.0 && 0.0 <= proby1gty2 <= 1.0
    @assert proby1lty2 + proby1gty2 < 1.0
    return proby1lty2, proby1gty2
end

function probresults_from_means(mean1::Float64, mean2::Float64, corr12::Float64)::Tuple{Float64, Float64}
#
# (Pr(Y1 < Y2), Pr(Y1 > Y2)) with
# Y1, Y2 ~ probscore_from_means(y1, y2; mean1, mean2, corr12).
#
    msg = "probresults_from_means(mean1 = $(mean1), mean2 = $(mean2), corr12 = $(corr12))"
    @assert 0.0 <= mean1 < Inf [msg]
    @assert 0.0 <= mean2 < Inf [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    @assert meansvalid(mean1, mean2, corr12)
    lambda1, lambda2, lambdax = meanstolambdas(mean1, mean2, corr12)
    proby1lty2, proby1gty2 = probresults_from_lambdas(lambda1, lambda2, lambdax)
    @assert 0.0 <= proby1lty2 <= 1.0 && 0.0 <= proby1gty2 <= 1.0
    @assert proby1lty2 + proby1gty2 < 1.0
    return proby1lty2, proby1gty2
end

# ==============================================================================
# Map [0.0..+Inf] <-> [-Inf..+Inf]
# to support positive-only variables in nlsolve().
# sqrt() included to "encourage" nlsove() to converge to low values.
# ==============================================================================

function positivetounbounded(x::Float64)::Float64
    @assert 0.0 <= x ["positivetounbounded(x = $(x))"]
    y = if (x < 1.0) log(x) + 1.0 else sqrt(x) end
    return y
end

@assert positivetounbounded(0.0) == -Inf ["positivetounbounded(0.0) == $(positivetounbounded(0.0)) != -Inf"]
@assert positivetounbounded(1.0) == 1.0 ["positivetounbounded(1.0) == $(positivetounbounded(1.0)) != 1.0"]
@assert positivetounbounded(Inf) == Inf ["positivetounbounded(Inf) == $(positivetounbounded(Inf)) != Inf"]

function unboundedtopositive(y::Float64)::Float64
    x = if (y < 1.0) exp(y - 1.0) else y^2 end
    @assert 0.0 <= x ["unboundedtopositive(y = $(y)) = $(x)"]
    return x
end

@assert unboundedtopositive(-Inf) == 0.0 ["unboundedtopositive(-Inf) == $(unboundedtopositive(-Inf)) != 0.0"]
@assert unboundedtopositive(1.0) == 1.0 ["unboundedtopositive(1.0) == $(unboundedtopositive(1.0)) != 1.0"]
@assert unboundedtopositive(Inf) == Inf ["unboundedtopositive(Inf) == $(unboundedtopositive(Inf)) != Inf"]

# ==============================================================================
# lambdas_from_probresults(proby1lty2, proby1yty2, lambdax) -> (lambda1, lambda2)
# ==============================================================================

function lambdas_from_probresults(proby1lty2::Float64, proby1gty2::Float64, lambdax::Float64, lambdatolerance::Float64)::Tuple{Float64, Float64}
    msg = "lambdas_from_probresults(proby1lty2 = $(proby1lty2), proby1gty2 = $(proby1gty2), lambdax = $(lambdax), lambdatolerance = $(lambdatolerance))"
    @assert 0.0 <= proby1lty2 [msg]
    @assert 0.0 <= proby1gty2 [msg]
    @assert 0.0 <= proby1lty2 + proby1gty2 < 1.0 [msg]
    @assert 0.0 <= lambdax < Inf [msg]
    @assert 0.0 < lambdatolerance [msg]
    if proby1lty2 > proby1gty2  # Ensure numerical symmetry
        lambda2, lambda1 = lambdas_from_probresults(proby1gty2, proby1lty2, lambdax, lambdatolerance)
    elseif proby1lty2 == proby1gty2 == 0.0
        lambda1, lambda2 = (0.0, 0.0)
    else
        key = (proby1lty2, proby1gty2, lambdax, lambdatolerance)
        lambda1, lambda2 = get(lambdas_from_probresults_cache, key, (-1.0, -1.0))
        if (lambda1, lambda2) == (-1.0, -1.0)
            probtolerance = max(eps(4.0), lambdatolerance/4)
            if proby1lty2 == proby1gty2
                # Solve probresults_from_lambdas(lambda, lambda, lambdax) == (proby1lty2, proby1gty2) for lambda
                function funequal(inputlambda::Vector{Float64})::Vector{Float64}
                    @assert length(inputlambda) == 1
                    estimatelambda = unboundedtopositive(inputlambda[1])
                    estimateproby1lty2, estimateproby1gty2 = probresults_from_lambdas(estimatelambda, estimatelambda, lambdax)
                    @assert estimateproby1lty2 == estimateproby1gty2 [msg * ": estimatelambda = $(estimatelambda), estimateproby1lty2 = $(estimateproby1lty2), estimateproby1gty2 =$(estimateproby1gty2)"]
                    return [estimateproby1lty2 - proby1lty2]
                end
                solution = nlsolve(funequal, [positivetounbounded(1.0)], xtol = lambdatolerance, ftol = probtolerance, show_trace = false)
                @assert converged(solution) [msg]
                lambda1, lambda2 = unboundedtopositive.((solution.zero[1], solution.zero[1]))
            elseif proby1lty2 == 0.0
                # Solve probresults_from_lambdas(lambda1, 0.0, lambdax) == (0.0, proby1gty2) for lambda1
                function fun1(inputlambda::Vector{Float64})::Vector{Float64}
                    @assert length(inputlambda) == 1
                    estimatelambda1 = unboundedtopositive(inputlambda[1])
                    estimateproby1lty2, estimateproby1gty2 = probresults_from_lambdas(estimatelambda1, 0.0, lambdax)
                    @assert estimateproby1lty2 == 0.0 [msg * ": estimatelambda1 = $(estimatelambda1), estimateproby1lty2 = $(estimateproby1lty2), estimateproby1gty2 = $(estimateproby1gty2)"]
                    return [estimateproby1gty2 - proby1gty2]
                end
                solution = nlsolve(fun1, [positivetounbounded(1.0)], xtol = lambdatolerance, ftol = probtolerance, show_trace = false)
                @assert converged(solution) [msg]
                lambda1, lambda2 = (unboundedtopositive(solution.zero[1]), 0.0)
            else
                # Solve probresults_from_lambdas(lambda1, lambda2, lambdax) == (proby1lty2, proby1gty2) for (lambda1, lambda2)
                function fun2(inputlambdas::Vector{Float64})::Vector{Float64}
                    @assert length(inputlambdas) == 2
                    estimatelambda1, estimatelambda2 = unboundedtopositive.((inputlambdas[1], inputlambdas[2]))
                    estimateproby1lty2, estimateproby1gty2 = probresults_from_lambdas(estimatelambda1, estimatelambda2, lambdax)
                    @assert !isnan(estimateproby1lty2) && !isnan(estimateproby1gty2) [msg * ": estimatelambda1 = $(estimatelambda1), estimatelambda2 = $(estimatelambda2), estimateproby1lty2 = $(estimateproby1lty2), estimateproby1gty2 = $(estimateproby1gty2)"]
                    return [estimateproby1lty2 - proby1lty2, estimateproby1gty2 - proby1gty2]
                end
                solution = nlsolve(fun2, positivetounbounded.([1.0, 1.0]), xtol = lambdatolerance, ftol = probtolerance, show_trace = false)
                @assert converged(solution) [msg]
                lambda1, lambda2 = unboundedtopositive.((solution.zero[1], solution.zero[2]))
            end
            global lambdas_from_probresults_cache[key] = (lambda1, lambda2)
        end
    end
    @assert 0.0 <= lambda1 < Inf && 0.0 <= lambda2 < Inf
    return lambda1, lambda2
end

# ==============================================================================
# Map [min..max] <-> [-Inf..+Inf] using logit and logistic (inverse logit) functions
# to support probability variables in nlsolve().
# ==============================================================================

function rangetounbounded(x::Float64, min::Float64, max::Float64)::Float64
    @assert min <= x <= max ["rangetounbounded(x = $(x), min = $(min), max = $(max))"]
    if min == x == max
        y = 0.0
    else
        scaledx = (x - min)/(max - min)
        y = log(scaledx/(1.0 - scaledx))
    end
    return y
end

@assert rangetounbounded(0.0, 0.0, 0.0) == 0.0 ["rangetounbounded(0.0, 0.0, 0.0) == $(rangetounbounded(0.0, 0.0, 0.0)) != 0.0"]
@assert rangetounbounded(0.0, 0.0, 0.5) == -Inf ["rangetounbounded(0.0, 0.0, 0.5) == $(rangetounbounded(0.0, 0.0, 0.5)) != -Inf"]
@assert rangetounbounded(0.25, 0.0, 0.5) == 0.0 ["rangetounbounded(0.25, 0.0, 0.5) == $(rangetounbounded(0.25, 0.0, 0.5)) != 0.0"]
@assert rangetounbounded(0.5, 0.0, 0.5) == Inf ["rangetounbounded(0.5, 0.0, 0.5) == $(rangetounbounded(0.5, 0.0, 0.5)) != Inf"]
@assert rangetounbounded(0.0, 0.0, 1.0) == -Inf ["rangetounbounded(0.0, 0.0, 1.0) == $(rangetounbounded(0.0, 0.0, 1.0)) != -Inf"]
@assert rangetounbounded(0.5, 0.0, 1.0) == 0.0 ["rangetounbounded(0.5, 0.0, 1.0) == $(rangetounbounded(0.5, 0.0, 1.0)) != 0.0"]
@assert rangetounbounded(1.0, 0.0, 1.0) == Inf ["rangetounbounded(1.0, 0.0, 1.0) == $(rangetounbounded(1.0, 0.0, 1.0)) != Inf"]
@assert rangetounbounded(0.0, 0.0, 2.0) == -Inf ["rangetounbounded(0.0, 0.0, 2.0) == $(rangetounbounded(0.0, 0.0, 2.0)) != -Inf"]
@assert rangetounbounded(1.0, 0.0, 2.0) == 0.0 ["rangetounbounded(1.0, 0.0, 2.0) == $(rangetounbounded(1.0, 0.0, 2.0)) != 0.0"]
@assert rangetounbounded(2.0, 0.0, 2.0) == Inf ["rangetounbounded(2.0, 0.0, 2.0) == $(rangetounbounded(2.0, 0.0, 2.0)) != Inf"]

@assert rangetounbounded(1.0, 1.0, 1.0) == 0.0 ["rangetounbounded(1.0, 1.0, 1.0) == $(rangetounbounded(1.0, 1.0, 1.0)) != 0.0"]
@assert rangetounbounded(1.0, 1.0, 1.5) == -Inf ["rangetounbounded(1.0, 1.0, 1.5) == $(rangetounbounded(1.0, 1.0, 1.5)) != -Inf"]
@assert rangetounbounded(1.25, 1.0, 1.5) == 0.0 ["rangetounbounded(1.25, 1.0, 1.5) == $(rangetounbounded(1.25, 1.0, 1.5)) != 0.0"]
@assert rangetounbounded(1.5, 1.0, 1.5) == Inf ["rangetounbounded(1.5, 1.0, 1.5) == $(rangetounbounded(1.5, 1.0, 1.5)) != Inf"]
@assert rangetounbounded(1.0, 1.0, 2.0) == -Inf ["rangetounbounded(1.0, 1.0, 2.0) == $(rangetounbounded(1.0, 1.0, 2.0)) != -Inf"]
@assert rangetounbounded(1.5, 1.0, 2.0) == 0.0 ["rangetounbounded(1.5, 1.0, 2.0) == $(rangetounbounded(1.5, 1.0, 2.0)) != 0.0"]
@assert rangetounbounded(2.0, 1.0, 2.0) == Inf ["rangetounbounded(2.0, 1.0, 2.0) == $(rangetounbounded(2.0, 1.0, 2.0)) != Inf"]
@assert rangetounbounded(1.0, 1.0, 3.0) == -Inf ["rangetounbounded(1.0, 1.0, 3.0) == $(rangetounbounded(1.0, 1.0, 3.0)) != -Inf"]
@assert rangetounbounded(2.0, 1.0, 3.0) == 0.0 ["rangetounbounded(2.0, 1.0, 3.0) == $(rangetounbounded(2.0, 1.0, 3.0)) != 0.0"]
@assert rangetounbounded(3.0, 1.0, 3.0) == Inf ["rangetounbounded(3.0, 1.0, 3.0) == $(rangetounbounded(3.0, 1.0, 3.0)) != Inf"]

@assert rangetounbounded(-1.0, -1.0, -1.0) == 0.0 ["rangetounbounded(-1.0, -1.0, -1.0) == $(rangetounbounded(-1.0, -1.0, -1.0)) != 0.0"]
@assert rangetounbounded(-1.0, -1.0, 1.0) == -Inf ["rangetounbounded(-1.0, -1.0, 1.0) == $(rangetounbounded(-1.0, -1.0, 1.0)) != -Inf"]
@assert rangetounbounded(0.0, -1.0, 1.0) == 0.0 ["rangetounbounded(0.0, -1.0, 1.0) == $(rangetounbounded(0.0, -1.0, 1.0)) != 0.0"]
@assert rangetounbounded(1.0, -1.0, 1.0) == Inf ["rangetounbounded(1.0, -1.0, 1.0) == $(rangetounbounded(1.0, -1.0, 1.0)) != Inf"]
@assert rangetounbounded(-1.0, -1.0, 1.0) == -Inf ["rangetounbounded(-1.0, -1.0, 1.0) == $(rangetounbounded(-1.0, -1.0, 1.0)) != -Inf"]
@assert rangetounbounded(-1.5, -2.0, -1.0) == 0.0 ["rangetounbounded(-1.5, -2.0, -1.0) == $(rangetounbounded(-1.5, -2.0, -1.0)) != 0.0"]
@assert rangetounbounded(-1.0, -2.0, -1.0) == Inf ["rangetounbounded(-1.0, -2.0, -1.0) == $(rangetounbounded(-1.0, -2.0, -1.0)) != Inf"]
@assert rangetounbounded(-2.0, -2.0, -1.0) == -Inf ["rangetounbounded(-2.0, -2.0, -1.0) == $(rangetounbounded(-2.0, -2.0, -1.0)) != -Inf"]
@assert rangetounbounded(-2.0, -4.0, 0.0) == 0.0 ["rangetounbounded(-2.0, -4.0, 0.0) == $(rangetounbounded(-2.0, -4.0, 0.0)) != 0.0"]
@assert rangetounbounded(0.0, -4.0, 0.0) == Inf ["rangetounbounded(0.0, -4.0, 0.0) == $(rangetounbounded(0.0, -4.0, 0.0)) != Inf"]

function unboundedtorange(y::Float64, min::Float64, max::Float64)::Float64
    @assert min <= max ["unboundedtorange(y = $(y), min = $(min), max = $(max))"]
    @assert min < max || y == 0.0
    x = if (min == max) min else min + (max - min)/(1.0 + exp(-y)) end
    @assert min <= x <= max ["unboundedtorange(y = $(y), min = $(min), max = $(max)) = $(x)"]
    return x
end

@assert unboundedtorange(0.0, 0.0, 0.0) == 0.0 ["unboundedtorange(0.0, 0.0, 0.0) == $(unboundedtorange(0.0, 0.0, 0.0)) != 0.0"]
@assert unboundedtorange(-Inf, 0.0, 0.5) == 0.0 ["unboundedtorange(-Inf, 0.0, 0.5) == $(unboundedtorange(-Inf, 0.0, 0.5)) != 0.0"]
@assert unboundedtorange(0.0, 0.0, 0.5) == 0.25 ["unboundedtorange(0.0, 0.0, 0.5) == $(unboundedtorange(0.0, 0.0, 0.5)) != 0.25"]
@assert unboundedtorange(Inf, 0.0, 0.5) == 0.5 ["unboundedtorange(Inf, 0.0, 0.5) == $(unboundedtorange(Inf, 0.0, 0.5)) != 0.5"]
@assert unboundedtorange(-Inf, 0.0, 1.0) == 0.0 ["unboundedtorange(-Inf, 0.0, 1.0) == $(unboundedtorange(-Inf, 0.0, 1.0)) != 0.0"]
@assert unboundedtorange(0.0, 0.0, 1.0) == 0.5 ["unboundedtorange(0.0, 0.0, 1.0) == $(unboundedtorange(0.0, 0.0, 1.0)) != 0.5"]
@assert unboundedtorange(Inf, 0.0, 1.0) == 1.0 ["unboundedtorange(Inf, 0.0, 1.0) == $(unboundedtorange(Inf, 0.0, 1.0)) != 1.0"]
@assert unboundedtorange(-Inf, 0.0, 2.0) == 0.0 ["unboundedtorange(-Inf, 0.0, 2.0) == $(unboundedtorange(-Inf, 0.0, 2.0)) != 0.0"]
@assert unboundedtorange(0.0, 0.0, 2.0) == 1.0 ["unboundedtorange(0.0, 0.0, 2.0) == $(unboundedtorange(0.0, 0.0, 2.0)) != 1.0"]
@assert unboundedtorange(Inf, 0.0, 2.0) == 2.0 ["unboundedtorange(Inf, 0.0, 2.0) == $(unboundedtorange(Inf, 0.0, 2.0)) != 2.0"]

@assert unboundedtorange(0.0, 1.0, 1.0) == 1.0 ["unboundedtorange(0.0, 1.0, 1.0) == $(unboundedtorange(0.0, 1.0, 1.0)) != 1.0"]
@assert unboundedtorange(-Inf, 1.0, 1.5) == 1.0 ["unboundedtorange(-Inf, 1.0, 1.5) == $(unboundedtorange(-Inf, 1.0, 1.5)) != 1.0"]
@assert unboundedtorange(0.0, 1.0, 1.5) == 1.25 ["unboundedtorange(0.0, 1.0, 1.5) == $(unboundedtorange(0.0, 1.0, 1.5)) != 1.25"]
@assert unboundedtorange(Inf, 1.0, 1.5) == 1.5 ["unboundedtorange(Inf, 1.0, 1.5) == $(unboundedtorange(Inf, 1.0, 1.5)) != 1.5"]
@assert unboundedtorange(-Inf, 1.0, 2.0) == 1.0 ["unboundedtorange(-Inf, 1.0, 2.0) == $(unboundedtorange(-Inf, 1.0, 2.0)) != 1.0"]
@assert unboundedtorange(0.0, 1.0, 2.0) == 1.5 ["unboundedtorange(0.0, 1.0, 2.0) == $(unboundedtorange(0.0, 1.0, 2.0)) != 1.5"]
@assert unboundedtorange(Inf, 1.0, 2.0) == 2.0 ["unboundedtorange(Inf, 1.0, 2.0) == $(unboundedtorange(Inf, 1.0, 2.0)) != 2.0"]
@assert unboundedtorange(-Inf, 1.0, 3.0) == 1.0 ["unboundedtorange(-Inf, 1.0, 3.0) == $(unboundedtorange(-Inf, 1.0, 3.0)) != 1.0"]
@assert unboundedtorange(0.0, 1.0, 3.0) == 2.0 ["unboundedtorange(0.0, 1.0, 3.0) == $(unboundedtorange(0.0, 1.0, 3.0)) != 2.0"]
@assert unboundedtorange(Inf, 1.0, 3.0) == 3.0 ["unboundedtorange(Inf, 1.0, 3.0) == $(unboundedtorange(Inf, 1.0, 3.0)) != 3.0"]

@assert unboundedtorange(0.0, -1.0, -1.0) == -1.0 ["unboundedtorange(0.0, -1.0, -1.0) == $(unboundedtorange(0.0, -1.0, -1.0)) != -1.0"]
@assert unboundedtorange(-Inf, -1.0, 1.0) == -1.0 ["unboundedtorange(-Inf, -1.0, 1.0) == $(unboundedtorange(-Inf, -1.0, 1.0)) != -1.0"]
@assert unboundedtorange(0.0, -1.0, 1.0) == 0.0 ["unboundedtorange(0.0, -1.0, 1.0) == $(unboundedtorange(0.0, -1.0, 1.0)) != 0.0"]
@assert unboundedtorange(Inf, -1.0, 1.0) == 1.0 ["unboundedtorange(Inf, -1.0, 1.0) == $(unboundedtorange(Inf, -1.0, 1.0)) != 1.0"]
@assert unboundedtorange(-Inf, -1.0, 1.0) == -1.0 ["unboundedtorange(-Inf, -1.0, 1.0) == $(unboundedtorange(-Inf, -1.0, 1.0)) != -1.0"]
@assert unboundedtorange(0.0, -2.0, -1.0) == -1.5 ["unboundedtorange(0.0, -2.0, -1.0) == $(unboundedtorange(0.0, -2.0, -1.0)) != -1.5"]
@assert unboundedtorange(Inf, -2.0, -1.0) == -1.0 ["unboundedtorange(Inf, -2.0, -1.0) == $(unboundedtorange(Inf, -2.0, -1.0)) != -1.0"]
@assert unboundedtorange(-Inf, -2.0, -1.0) == -2.0 ["unboundedtorange(-Inf, -2.0, -1.0) == $(unboundedtorange(-Inf, -2.0, -1.0)) != -2.0"]
@assert unboundedtorange(0.0, -4.0, 0.0) == -2.0 ["unboundedtorange(0.0, -4.0, 0.0) == $(unboundedtorange(0.0, -4.0, 0.0)) != -2.0"]
@assert unboundedtorange(Inf, -4.0, 0.0) == 0.0 ["unboundedtorange(Inf, -4.0, 0.0) == $(unboundedtorange(Inf, -4.0, 0.0)) != 0.0"]

# ==============================================================================
# Map (mean1, mean2) <-> ([-Inf..+Inf, -Inf..+Inf]) for a given corr12
# to support bounded means in nlsolve().
# ==============================================================================

function meanstounbounded(mean1::Float64, mean2::Float64, corr12::Float64)::Tuple{Float64, Float64}
    @assert 0.0 <= mean1 && 0.0 <= mean2 && 0.0 <= corr12 <= 1.0 && meansvalid(mean1, mean2, corr12) ["meanstounbounded(mean1 = $(mean1), mean2 = $(mean2), corr12 = $(corr12))"]
    magnitude = mean1 + mean2  # Range: 0..Inf
    ratio = if (mean1 == mean2) 0.0 elseif (mean1 <= mean2)  mean1/mean2 - 1.0 else 1.0 - mean2/mean1 end  # Range: (corr12^2 - 1.0)..(1.0 - corr12^2)
    return (positivetounbounded(magnitude), rangetounbounded(ratio, corr12^2 - 1.0, 1.0 - corr12^2))
end

@assert meanstounbounded(0.0, 0.0, 0.0) == (-Inf, 0.0) ["meanstounbounded(0.0, 0.0, 0.0) = $(meanstounbounded(0.0, 0.0, 0.0)) != (-Inf, 0.0)"]
@assert meanstounbounded(0.0, 0.0, 0.5) == (-Inf, 0.0) ["meanstounbounded(0.0, 0.0, 0.5) = $(meanstounbounded(0.0, 0.0, 0.5)) != (-Inf, 0.0)"]
@assert meanstounbounded(0.0, 0.0, 1.0) == (-Inf, 0.0) ["meanstounbounded(0.0, 0.0, 1.0) = $(meanstounbounded(0.0, 0.0, 1.0)) != (-Inf, 0.0)"]

@assert meanstounbounded(1.0, 0.0, 0.0) == (1.0, Inf) ["meanstounbounded(1.0, 0.0, 0.0) = $(meanstounbounded(1.0, 0.0, 0.0)) != (1.0, Inf)"]
@assert meanstounbounded(0.0, 1.0, 0.0) == (1.0, -Inf) ["meanstounbounded(0.0, 1.0, 0.0) = $(meanstounbounded(1.0, 0.0, 0.0)) != (1.0, -Inf)"]

@assert round.(meanstounbounded(1.0, 3.0, 0.0), digits = 4) == round.((2.0, log(1/5)), digits = 4) ["meanstounbounded(1.0, 3.0, 0.0) = $(meanstounbounded(1.0, 3.0, 0.0)) != (2.0, log(1/5))"]
@assert round.(meanstounbounded(3.0, 1.0, 0.0), digits = 4) == round.((2.0, log(5)), digits = 4) ["meanstounbounded(3.0, 1.0, 0.0) = $(meanstounbounded(3.0, 1.0, 0.0)) != (2.0, log(5))"]
@assert round.(meanstounbounded(1.0, 3.0, sqrt(1/6)), digits = 4) == round.((2.0, log(1/9)), digits = 4) ["meanstounbounded(1.0, 3.0, sqrt(1/6)) = $(meanstounbounded(1.0, 3.0, sqrt(1/6))) != (2.0, log(1/9))"]
@assert round.(meanstounbounded(3.0, 1.0, sqrt(1/6)), digits = 4) == round.((2.0, log(9)), digits = 4) ["meanstounbounded(3.0, 1.0, sqrt(1/6)) = $(meanstounbounded(3.0, 1.0, sqrt(1/6))) != (2.0, log(9))"]
@assert meanstounbounded(1.0, 3.0, sqrt(1/3)) == (2.0, -Inf) ["meanstounbounded(1.0, 3.0, sqrt(1/3)) = $(meanstounbounded(1.0, 3.0, sqrt(1/3))) != (2.0, -Inf)"]
@assert meanstounbounded(3.0, 1.0, sqrt(1/3)) == (2.0, Inf) ["meanstounbounded(3.0, 1.0, sqrt(1/3)) = $(meanstounbounded(3.0, 1.0, sqrt(1/3))) != (2.0, Inf)"]

@assert meanstounbounded(2.0, 2.0, 0.0) == (2.0, 0.0) ["meanstounbounded(2.0, 2.0, 0.0) = $(meanstounbounded(2.0, 2.0, 0.0)) != (2.0, 0.0)"]
@assert meanstounbounded(2.0, 2.0, 0.5) == (2.0, 0.0) ["meanstounbounded(2.0, 2.0, 0.5) = $(meanstounbounded(2.0, 2.0, 0.5)) != (2.0, 0.0)"]
@assert meanstounbounded(2.0, 2.0, 1.0) == (2.0, 0.0) ["meanstounbounded(2.0, 2.0, 1.0) = $(meanstounbounded(2.0, 2.0, 1.0)) != (2.0, 0.0)"]

function unboundedtomeans(y1::Float64, y2::Float64, corr12::Float64)::Tuple{Float64, Float64}
    @assert 0.0 <= corr12 <= 1.0 ["unboundedtomeans(y1 = $(y1), y2 = $(y2), corr12 = $(corr12))"]
    magnitude = unboundedtopositive(y1)
    ratio = unboundedtorange(y2, corr12^2 - 1.0, 1.0 - corr12^2)
    if ratio <= 0
        mean2 = magnitude/(ratio + 2.0)
        mean1 = magnitude - mean2
    else
        mean1 = magnitude/(2.0 - ratio)
        mean2 = magnitude - mean1
    end
    @assert 0.0 <= mean1 && 0.0 <= mean2 ["unboundedtomeans(y1 = $(y1), y2 = $(y2), corr12 = $(corr12)) = ($(mean1), $(mean2))"]
    if !(meansvalid(mean1, mean2, corr12))  # due to numeric precision rounding
        mean1 = if (mean1 < mean2) minsmallermean(mean2, corr12) else maxlargermean(mean2, corr12) end
    end
    return (mean1, mean2)
end

@assert unboundedtomeans(-Inf, 0.0, 0.0) == (0.0, 0.0) ["unboundedtomeans(-Inf, 0.0, 0.0) == $(unboundedtomeans(-Inf, 0.0, 0.0)) != (0.0, 0.0)"]
@assert unboundedtomeans(-Inf, 0.0, 0.5) == (0.0, 0.0) ["unboundedtomeans(-Inf, 0.0, 0.5) == $(unboundedtomeans(-Inf, 0.0, 0.5)) != (0.0, 0.0)"]
@assert unboundedtomeans(-Inf, 0.0, 1.0) == (0.0, 0.0) ["unboundedtomeans(-Inf, 0.0, 1.0) == $(unboundedtomeans(-Inf, 0.0, 1.0)) != (0.0, 0.0)"]

@assert unboundedtomeans(1.0, Inf, 0.0) == (1.0, 0.0) ["unboundedtomeans(1.0, Inf, 0.0) == $(unboundedtomeans(1.0, Inf, 0.0)) != (1.0, 0.0)"]
@assert unboundedtomeans(1.0, -Inf, 0.0) == (0.0, 1.0) ["unboundedtomeans(1.0, -Inf, 0.0) == $(unboundedtomeans(1.0, -Inf, 0.0)) != (0.0, 1.0)"]

@assert round.(unboundedtomeans(2.0, log(1/5), 0.0), digits = 4) == (1.0, 3.0) ["unboundedtomeans(2.0, log(1/5), 0.0) == $(unboundedtomeans(2.0, log(1/5), 0.0)) != (1.0, 3.0)"]
@assert round.(unboundedtomeans(2.0, log(5), 0.0), digits = 4) == (3.0, 1.0) ["unboundedtomeans(2.0, log(5), 0.0) == $(unboundedtomeans(2.0, log(5), 0.0)) != (3.0, 1.0)"]
@assert round.(unboundedtomeans(2.0, log(1/9), sqrt(1/6)), digits = 4) == (1.0, 3.0) ["unboundedtomeans(2.0, log(1/9), sqrt(1/6)) == $(unboundedtomeans(2.0, log(1/9), sqrt(1/6))) != (1.0, 3.0)"]
@assert round.(unboundedtomeans(2.0, log(9), sqrt(1/6)), digits = 4) == (3.0, 1.0) ["unboundedtomeans(2.0, log(9), sqrt(1/6)) == $(unboundedtomeans(2.0, log(9), sqrt(1/6))) != (3.0, 1.0)"]
@assert round.(unboundedtomeans(2.0, -Inf, sqrt(1/3)), digits = 4) == (1.0, 3.0) ["unboundedtomeans(2.0, -Inf, sqrt(1/3)) == $(unboundedtomeans(2.0, -Inf, sqrt(1/3))) != (1.0, 3.0)"]
@assert round.(unboundedtomeans(2.0, Inf, sqrt(1/3)), digits = 4) == (3.0, 1.0) ["unboundedtomeans(2.0, Inf, sqrt(1/3)) == $(unboundedtomeans(2.0, Inf, sqrt(1/3))) != (3.0, 1.0)"]

@assert unboundedtomeans(2.0, 0.0, 0.0) == (2.0, 2.0) ["unboundedtomeans(2.0, 0.0, 0.0) == $(unboundedtomeans(2.0, 0.0, 0.0)) != (2.0, 2.0)"]
@assert unboundedtomeans(2.0, 0.0, 0.5) == (2.0, 2.0) ["unboundedtomeans(2.0, 0.0, 0.5) == $(unboundedtomeans(2.0, 0.0, 0.5)) != (2.0, 2.0)"]
@assert unboundedtomeans(2.0, 0.0, 1.0) == (2.0, 2.0) ["unboundedtomeans(2.0, 0.0, 1.0) == $(unboundedtomeans(2.0, 0.0, 1.0)) != (2.0, 2.0)"]

# ==============================================================================
# means_from_probresults(proby1lty2, proby1yty2, corr12) -> (mean1, mean2)
# ==============================================================================

function means_from_probresults(proby1lty2::Float64, proby1gty2::Float64, corr12::Float64, meantolerance::Float64)::Tuple{Float64, Float64}
    msg = "means_from_probresults(proby1lty2 = $(proby1lty2), proby1gty2 = $(proby1gty2), corr12 = $(corr12), meantolerance = $(meantolerance))"
    @assert 0.0 <= proby1lty2 [msg]
    @assert 0.0 <= proby1gty2 [msg]
    @assert 0.0 <= proby1lty2 + proby1gty2 < 1.0 [msg]
    @assert 0.0 <= corr12 <= 1.0 [msg]
    @assert !(corr12 == 1.0) || (proby1lty2 == 0.0 && proby1gty2 == 0.0) [msg]  # (corr12 == 1.0) => certain draw
    @assert 0.0 < meantolerance [msg]
    if proby1lty2 > proby1gty2  # Ensure numerical symmetry
        mean2, mean1 = means_from_probresults(proby1gty2, proby1lty2, corr12, meantolerance)
    elseif proby1lty2 == proby1gty2 == 0.0
        mean1, mean2 = (0.0, 0.0)  # for all corrAB (including 1), limit (meanA, meanB) == 0 as (probwinB, probwinA) -> (0, 0)
    else
        key = (proby1lty2, proby1gty2, corr12, meantolerance)
        mean1, mean2 = get(means_from_probresults_cache, key, (-1.0, -1.0))
        if (mean1, mean2) == (-1.0, -1.0)
            probtolerance = max(eps(4.0), meantolerance/4)
            if proby1lty2 == proby1gty2
                # Solve probresults_from_means(mean, mean, corr12) == (proby1lty2, proby1gty2) for mean
                function funequal(inputmean::Vector{Float64})::Vector{Float64}
                    @assert length(inputmean) == 1
                    estimatemean = unboundedtopositive(inputmean[1])
                    estimateproby1lty2, estimateproby1gty2 = probresults_from_means(estimatemean, estimatemean, corr12)
                    @assert estimateproby1lty2 == estimateproby1gty2 [msg * ": estimatemean = $(estimatemean), estimateproby1lty2 = $(estimateproby1lty2), estimateproby1gty2 =$(estimateproby1gty2)"]
                    return [estimateproby1lty2 - proby1lty2]
                end
                solution = nlsolve(funequal, [positivetounbounded(1.0)], xtol = meantolerance, ftol = probtolerance, show_trace = false)
                @assert converged(solution) [msg]
                mean1, mean2 = unboundedtopositive.((solution.zero[1], solution.zero[1]))
            elseif proby1lty2 == 0.0 && corr12 == 0.0
                # Solve probresults_from_means(mean1, 0.0, corr12) == (0.0, proby1gty2) for mean1
                function fun1(inputmean::Vector{Float64})::Vector{Float64}
                    @assert length(inputmean) == 1
                    estimatemean1 = unboundedtopositive(inputmean[1])
                    estimateproby1lty2, estimateproby1gty2 = probresults_from_means(estimatemean1, 0.0, corr12)
                    @assert estimateproby1lty2 == 0.0 [msg * ": estimatemean1 = $(estimatemean1), estimateproby1lty2 = $(estimateproby1lty2), estimateproby1gty2 = $(estimateproby1gty2)"]
                    return [estimateproby1gty2 - proby1gty2]
                end
                solution = nlsolve(fun1, [positivetounbounded(1.0)], xtol = meantolerance, ftol = probtolerance, show_trace = false)
                @assert converged(solution) [msg]
                mean1, mean2 = (unboundedtopositive(solution.zero[1]), 0.0)
            else
                # Solve probresults_from_means(mean1, mean2, corr12) == (proby1lty2, proby1gty2) for (mean1, mean2)
                function fun2(inputmeans::Vector{Float64})::Vector{Float64}
                    @assert length(inputmeans) == 2
                    estimatemean1, estimatemean2 = unboundedtomeans(inputmeans[1], inputmeans[2], corr12)
                    estimateproby1lty2, estimateproby1gty2 = probresults_from_means(estimatemean1, estimatemean2, corr12)
                    @assert !isnan(estimateproby1lty2) && !isnan(estimateproby1gty2) [msg * ": estimatemean1 = $(estimatemean1), estimatemean2 = $(estimatemean2), estimateproby1lty2 = $(estimateproby1lty2), estimateproby1gty2 = $(estimateproby1gty2)"]
                    return [estimateproby1lty2 - proby1lty2, estimateproby1gty2 - proby1gty2]
                end
                solution = nlsolve(fun2, collect(meanstounbounded.(1.0, 1.0, corr12)), xtol = meantolerance, ftol = probtolerance, show_trace = false)
                @assert converged(solution) [msg]
                mean1, mean2 = unboundedtomeans.(solution.zero[1], solution.zero[2], corr12)
            end
            global means_from_probresults_cache[key] = (mean1, mean2)
        end
    end
    @assert 0.0 <= mean1 < Inf && 0.0 <= mean2 < Inf
    return mean1, mean2
end

# ==============================================================================
# Map [0.0..max] <-> [-Inf..+Inf] using logit and logistic (inverse logit) functions
# to support probability variables in nlsolve().
# sqrt() included to "encourage" nlsolve() to converge to low values.
# ==============================================================================

function boundedtounbounded(x::Float64, max::Float64)::Float64
    @assert 0.0 <= x <= max ["boundedtounbounded(x = $(x), max = $(max))"]
    scaledx = sqrt(x/max)
    y = log(scaledx/(1.0 - scaledx))
    return y
end

@assert boundedtounbounded(0.0, 0.5) == -Inf ["boundedtounbounded(0.0, 0.5) == $(boundedtounbounded(0.0, 0.5)) != -Inf"]
@assert boundedtounbounded(0.125, 0.5) == 0.0 ["boundedtounbounded(0.125, 0.5) == $(boundedtounbounded(0.125, 0.5)) != 0.0"]
@assert boundedtounbounded(0.5, 0.5) == Inf ["boundedtounbounded(0.5, 0.5) == $(boundedtounbounded(0.5, 0.5)) != Inf"]
@assert boundedtounbounded(0.0, 1.0) == -Inf ["boundedtounbounded(0.0, 1.0) == $(boundedtounbounded(0.0, 1.0)) != -Inf"]
@assert boundedtounbounded(0.25, 1.0) == 0.0 ["boundedtounbounded(0.25, 1.0) == $(boundedtounbounded(0.25, 1.0)) != 0.0"]
@assert boundedtounbounded(1.0, 1.0) == Inf ["boundedtounbounded(1.0, 1.0) == $(boundedtounbounded(1.0, 1.0)) != Inf"]
@assert boundedtounbounded(0.0, 2.0) == -Inf ["boundedtounbounded(0.0, 2.0) == $(boundedtounbounded(0.0, 2.0)) != -Inf"]
@assert boundedtounbounded(0.5, 2.0) == 0.0 ["boundedtounbounded(1.0, 2.0) == $(boundedtounbounded(0.5, 2.0)) != 0.0"]
@assert boundedtounbounded(2.0, 2.0) == Inf ["boundedtounbounded(2.0, 2.0) == $(boundedtounbounded(2.0, 2.0)) != Inf"]

function unboundedtobounded(y::Float64, max::Float64)::Float64
    @assert 0.0 <= max ["unboundedtobounded(y = $(y), max = $(max))"]
    x = max/(1.0 + exp(-y))^2
    @assert 0.0 <= x <= max ["unboundedtobounded(y = $(y), max = $(max)) = $(x)"]
    return x
end

@assert unboundedtobounded(-Inf, 0.5) == 0.0 ["unboundedtobounded(-Inf, 0.5) == $(unboundedtobounded(-Inf, 0.5)) != 0.0"]
@assert unboundedtobounded(0.0, 0.5) == 0.125 ["unboundedtobounded(0.0, 0.5) == $(unboundedtobounded(0.0, 0.5)) != 0.125"]
@assert unboundedtobounded(Inf, 0.5) == 0.5 ["unboundedtobounded(Inf, 0.5) == $(unboundedtobounded(Inf, 0.5)) != 0.5"]
@assert unboundedtobounded(-Inf, 1.0) == 0.0 ["unboundedtobounded(-Inf, 1.0) == $(unboundedtobounded(-Inf, 1.0)) != 0.0"]
@assert unboundedtobounded(0.0, 1.0) == 0.25 ["unboundedtobounded(0.0, 1.0) == $(unboundedtobounded(0.0, 1.0)) != 0.25"]
@assert unboundedtobounded(Inf, 1.0) == 1.0 ["unboundedtobounded(Inf, 1.0) == $(unboundedtobounded(Inf, 1.0)) != 1.0"]
@assert unboundedtobounded(-Inf, 2.0) == 0.0 ["unboundedtobounded(-Inf, 2.0) == $(unboundedtobounded(-Inf, 2.0)) != 0.0"]
@assert unboundedtobounded(0.0, 2.0) == 0.5 ["unboundedtobounded(0.0, 2.0) == $(unboundedtobounded(0.0, 2.0)) != 0.5"]
@assert unboundedtobounded(Inf, 2.0) == 2.0 ["unboundedtobounded(Inf, 2.0) == $(unboundedtobounded(Inf, 2.0)) != 2.0"]

# ==============================================================================
# proby1lty2_from_proby1gty2andlambda1(Pr(Y1 > Y2), lambda1, lambdax) -> Pr(Y1 < Y2)
# proby1gty2_from_proby1lty2andlambda2(Pr(Y1 < Y2), lambda2, lambdax) -> Pr(Y1 > Y2)
# proby1gty2_from_proby1lty2andlambda1(Pr(Y1 < Y2), lambda1, lambdax) -> Pr(Y1 > Y2)
# proby1lty2_from_proby1gty2andlambda2(Pr(Y1 > Y2), lambda2, lambdax) -> Pr(Y1 < Y2)
# ==============================================================================

function proby1lty2_from_proby1gty2andlambda1(proby1gty2::Float64, lambda1::Float64, lambdax::Float64, probtolerance::Float64)::Float64  # Can be NaN
# proby1lty2 such that there exists lambda2 where:
# Y1, Y2 ~ probscore_from_lambdas(y1, y2; lambda1, lambda2, lambdax) with
# Pr(Y1 < Y2) = proby1lty2 and Pr(Y1 > Y2) = proby1gty2.
# NaN if this does not exist.
# (From symmetry can also be used as proby1gty2_from_proby1lty2andlambda2())
    msg = "proby1lty2_from_proby1gty2andlambda1(proby1gty2 = $(proby1gty2), lambda1 = $(lambda1), lambdax = $(lambdax), probtolerance = $(probtolerance))"
    @assert 0.0 <= proby1gty2 < 1.0 [msg]
    @assert 0 < lambda1 < Inf [msg]
    @assert 0 <= lambdax < Inf [msg]
    @assert 0 <= probtolerance [msg]
    key = (proby1gty2, lambda1, lambdax, probtolerance)
    proby1lty2 = get(proby1lty2_from_proby1gty2andlambda1_cache, key, -1.0)
    if proby1lty2 == -1.0
        lambdatolerance = max(eps(4.0), probtolerance/4)
        zeroproblambda1, notusedlambda2 = lambdas_from_probresults(0.0, proby1gty2, lambdax, lambdatolerance)
        if lambda1 < zeroproblambda1
            proby1lty2 = NaN
        elseif lambda1 == zeroproblambda1
            proby1lty2 = 0.0
        elseif proby1gty2 == 0.0
            proby1lty2 = 1.0
        else
            function fun(inputprob::Vector{Float64})::Vector{Float64}
                @assert length(inputprob) == 1
                estimateproby1lty2 = unboundedtobounded(inputprob[1], 1.0 - proby1gty2)
                @assert estimateproby1lty2 + proby1gty2 < 1.0
                estimatelambda1, notusedlambda2 = lambdas_from_probresults(estimateproby1lty2, proby1gty2, lambdax, lambdatolerance)
                return [estimatelambda1 - lambda1]
            end
            solution = nlsolve(fun, [boundedtounbounded((1.0 - proby1gty2)/2, 1.0 - proby1gty2)], xtol = probtolerance, ftol = lambdatolerance, show_trace = false)
            @assert converged(solution) [msg]
            proby1lty2 = min(1.0 - proby1gty2, unboundedtobounded(solution.zero[1], 1.0 - proby1gty2))
        end
        global proby1lty2_from_proby1gty2andlambda1_cache[key] = proby1lty2
    end
    @assert isnan(proby1lty2) || 0.0 <= proby1lty2 <= 1.0
    return proby1lty2
end

function proby1gty2_from_proby1lty2andlambda1(proby1lty2::Float64, lambda1::Float64, lambdax::Float64, probtolerance::Float64)::Float64  # Never NaN
# proby1gty2 such that there exists lambda2 where:
# Y1, Y2 ~ probscore_from_lambdas(y1, y2; lambda1, lambda2, lambdax) with
# Pr(Y1 < Y2) = proby1lty2 and Pr(Y1 > Y2) = proby1gty2.
# Always exists.
# (From symmetry, can also be used as proby1lty2_from_proby1gty2andlambda2())
    msg = "proby1gty2_from_proby1lty2andlambda1(proby1lty2 = $(proby1lty2), lambda1 = $(lambda1), lambdax = $(lambdax), probtolerance = $(probtolerance))"
    @assert 0.0 <= proby1lty2 <= 1.0 [msg]
    @assert 0 < lambda1 < Inf [msg]
    @assert 0 <= lambdax < Inf [msg]
    @assert 0 <= probtolerance [msg]
    if proby1lty2 == 1.0
        proby1gty2 = 0.0
    else
        key = (proby1lty2, lambda1, lambdax, probtolerance)
        proby1gty2 = get(proby1gty2_from_proby1lty2andlambda1_cache, key, -1.0)
        if proby1gty2 == -1.0
            lambdatolerance = max(eps(4.0), probtolerance/4)
            function fun(inputprob::Vector{Float64})::Vector{Float64}
                @assert length(inputprob) == 1
                estimateproby1gty2 = unboundedtobounded(inputprob[1], 1.0 - proby1lty2)
                @assert estimateproby1gty2 + proby1lty2 < 1.0
                estimatelambda1, notusedlambda2 = lambdas_from_probresults(proby1lty2, estimateproby1gty2, lambdax, lambdatolerance)
                return [estimatelambda1 - lambda1]
            end
            solution = nlsolve(fun, [boundedtounbounded((1.0 - proby1lty2)/2, 1.0 - proby1lty2)], xtol = probtolerance, ftol = lambdatolerance, show_trace = false)
            @assert converged(solution) [msg]
            proby1gty2 = min(1.0 - proby1lty2, unboundedtobounded(solution.zero[1], 1.0 - proby1lty2))
            global proby1gty2_from_proby1lty2andlambda1_cache[key] = proby1gty2
        end
    end
    @assert 0.0 <= proby1gty2 < 1.0
    return proby1gty2
end

# ==============================================================================
# proby1lty2_from_proby1gty2andmean1(Pr(Y1 > Y2), mean1, corr12) -> Pr(Y1 < Y2)
# proby1gty2_from_proby1lty2andmean2(Pr(Y1 < Y2), mean2, corr12) -> Pr(Y1 > Y2)
# proby1gty2_from_proby1lty2andmean1(Pr(Y1 < Y2), mean1, corr12) -> Pr(Y1 > Y2)
# proby1lty2_from_proby1gty2andmean2(Pr(Y1 > Y2), mean2, corr12) -> Pr(Y1 < Y2)
# ==============================================================================

function proby1lty2_from_proby1gty2andmean1(proby1gty2::Float64, mean1::Float64, corr12::Float64, probtolerance::Float64)::Float64  # Can be NaN
# proby1lty2 such that there exists mean2 where:
# Y1, Y2 ~ probscore_from_means(y1, y2; mean1, mean2, corr12) with
# Pr(Y1 < Y2) = proby1lty2 and Pr(Y1 > Y2) = proby1gty2.
# NaN if this does not exist.
# (From symmetry can also be used as proby1gty2_from_proby1lty2andmean2())
    msg = "proby1lty2_from_proby1gty2andmean1(proby1gty2 = $(proby1gty2), mean1 = $(mean1), corr12 = $(corr12), probtolerance = $(probtolerance))"
    @assert 0.0 <= proby1gty2 < 1.0 [msg]
    @assert 0 <= mean1 < Inf [msg]
    @assert 0 <= corr12 <= 1.0 [msg]
    @assert 0 <= probtolerance [msg]
    if corr12 == 1.0
        proby1lty2 = if (proby1gty2 == 0.0) 0.0 else NaN end
    else
        key = (proby1gty2, mean1, corr12, probtolerance)
        proby1lty2 = get(proby1lty2_from_proby1gty2andmean1_cache, key, -1.0)
        if proby1lty2 == -1.0
            meantolerance = max(eps(4.0), probtolerance/4)
            zeroprobmean1, notusedmean2 = means_from_probresults(0.0, proby1gty2, corr12, meantolerance)
            if mean1 < zeroprobmean1
                proby1lty2 = NaN
            elseif mean1 == zeroprobmean1
                proby1lty2 = 0.0
            elseif proby1gty2 == 0.0 && corr12 == 0.0
                proby1lty2 = 1.0
            else
                function fun(inputprob::Vector{Float64})::Vector{Float64}
                    @assert length(inputprob) == 1
                    estimateproby1lty2 = unboundedtobounded(inputprob[1], 1.0 - proby1gty2)
                    @assert estimateproby1lty2 + proby1gty2 < 1.0
                    estimatemean1, notusedmean2 = means_from_probresults(estimateproby1lty2, proby1gty2, corr12, meantolerance)
                    return [estimatemean1 - mean1]
                end
                solution = nlsolve(fun, [boundedtounbounded((1.0 - proby1gty2)/2, 1.0 - proby1gty2)], xtol = probtolerance, ftol = meantolerance, show_trace = false)
                @assert converged(solution) [msg]
                proby1lty2 = min(1.0 - proby1gty2, unboundedtobounded(solution.zero[1], 1.0 - proby1gty2))
            end
            global proby1lty2_from_proby1gty2andmean1_cache[key] = proby1lty2
        end
    end
    @assert isnan(proby1lty2) || 0.0 <= proby1lty2 <= 1.0
    return proby1lty2
end

function proby1gty2_from_proby1lty2andmean1(proby1lty2::Float64, mean1::Float64, corr12::Float64, probtolerance::Float64)::Float64  # Can be NaN
# proby1gty2 such that there exists mean2 where:
# Y1, Y2 ~ probscore_from_means(y1, y2; mean1, mean2, corr12) with
# Pr(Y1 > Y2) = proby1gty2 and Pr(Y1 < Y2) = proby1lty2.
# NaN if this does not exist.
# (From symmetry can also be used as proby1lty2_from_proby1gty2andmean2().)
    msg = "proby1gty2_from_proby1lty2andmean1(proby1lty2 = $(proby1lty2), mean1 = $(mean1), corr12 = $(corr12), probtolerance = $(probtolerance))"
    @assert 0.0 <= proby1lty2 < 1.0 [msg]
    @assert 0 <= mean1 < Inf [msg]
    @assert 0 <= corr12 <= 1.0 [msg]
    @assert 0 <= probtolerance [msg]
    if corr12 == 1.0
        proby1gty2 = if (proby1lty2 == 0.0) 0.0 else NaN end
    else
        key = (proby1lty2, mean1, corr12, probtolerance)
        proby1gty2 = get(proby1gty2_from_proby1lty2andmean1_cache, key, -1.0)
        if proby1gty2 == -1.0
            meantolerance = max(eps(4.0), probtolerance/4)
            zeroprobmean1, notusedmean2 = means_from_probresults(proby1lty2, 0.0, corr12, meantolerance)
            if mean1 < zeroprobmean1
                proby1gty2 = NaN
            elseif mean1 == zeroprobmean1
                proby1gty2 = 0.0
            else
                function fun(inputprob::Vector{Float64})::Vector{Float64}
                    @assert length(inputprob) == 1
                    estimateproby1gty2 = unboundedtobounded(inputprob[1], 1.0 - proby1lty2)
                    @assert estimateproby1gty2 + proby1lty2 < 1.0
                    estimatemean1, notusedmean2 = means_from_probresults(proby1lty2, estimateproby1gty2, corr12, meantolerance)
                    return [estimatemean1 - mean1]
                end
                solution = nlsolve(fun, [boundedtounbounded((1.0 - proby1lty2)/2, 1.0 - proby1lty2)], xtol = probtolerance, ftol = meantolerance, show_trace = false)
                @assert converged(solution) [msg]
                proby1gty2 = min(1.0 - proby1lty2, unboundedtobounded(solution.zero[1], 1.0 - proby1lty2))
            end
            global proby1gty2_from_proby1lty2andmean1_cache[key] = proby1gty2
        end
    end
    @assert isnan(proby1gty2) || 0.0 <= proby1gty2 < 1.0
    return proby1gty2
end

# ==============================================================================
end
