module UtilTests
# ==============================================================================
# UtilTests
#
# Unit tests for functions exported by Util.jl.
# ==============================================================================

using Test

using Util

# ==============================================================================
# Test Points3D
# ==============================================================================

function test_Points3D_basics()::Nothing
    println("Start test_Points3D_basics")
    points = Points3D([1.0], [2.0], [3.0])
    @test length(points) == 1
    @test points.xs == [1.0]
    @test points.ys == [2.0]
    @test points.zs == [3.0]
    pushfirst!(points, 4.0, 5.0, 6.0)
    @test length(points) == 2
    @test points.xs == [4.0, 1.0]
    @test points.ys == [5.0, 2.0]
    @test points.zs == [6.0, 3.0]
    push!(points, 7.0, 8.0, 9.0)
    @test length(points) == 3
    @test points.xs == [4.0, 1.0, 7.0]
    @test points.ys == [5.0, 2.0, 8.0]
    @test points.zs == [6.0, 3.0, 9.0]
    println("End test_Points3D_basics")
end

function test_Points3D_sorted()::Nothing
    println("Start test_Points3D_sorted")
    unsorted = Points3D([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 1.0, 3.0])
    sorted_on_xs = sorted(unsorted, unsorted.xs)
    @test length(sorted_on_xs) == Util.length(unsorted)
    @test sorted_on_xs.xs == unsorted.xs
    @test sorted_on_xs.ys == unsorted.ys
    @test sorted_on_xs.zs == unsorted.zs
    sorted_on_ys = sorted(unsorted, unsorted.ys)
    @test length(sorted_on_ys) == length(unsorted)
    @test sorted_on_ys.xs == reverse(unsorted.xs)
    @test sorted_on_ys.ys == reverse(unsorted.ys)
    @test sorted_on_ys.zs == reverse(unsorted.zs)
    println("End test_Points3D_sorted")
end

# ==============================================================================
# Main program
# ==============================================================================

function main()::Nothing
    println("Start main")
    test_Points3D_basics()
    test_Points3D_sorted()
    println("End main")
end

main()

# ==============================================================================
end
