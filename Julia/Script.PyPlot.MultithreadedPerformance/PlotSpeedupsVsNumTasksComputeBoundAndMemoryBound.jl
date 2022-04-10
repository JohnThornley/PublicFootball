module PlotSpeedupsVsNumTasksComputeBoundAndMemoryBound
# ==============================================================================
# Plot chart of multhithreaded speedups vs number of tasks for
# compute-bound and memory-bound workloads.
#
# For discussion of Julia multithreading see:
# https://julialang.org/blog/2019/07/multithreading/
# https://proceedings.juliacon.org/papers/10.21105/jcon.00054
# ==============================================================================

using Statistics
using Base.Threads
import PyPlot; const plt = PyPlot  # See: https://github.com/JuliaPy/PyPlot.jl
using PyPlotSetup; setuppyplot2d!(plt)
using LaTeXStrings  # needed for L notation

# ==============================================================================
# Parameterization
# ==============================================================================

NUM_COMPUTE_ITERATIONS = 10_000_000_000
NUM_MEMORY_ITEMS = 5_000_000_000

NUM_OPERATINGSYSTEM_THREADS = Threads.nthreads()
NUM_TIMING_TRIALS = 10
MAX_TASKS = NUM_OPERATINGSYSTEM_THREADS*4
MAX_SPEEDUP = NUM_OPERATINGSYSTEM_THREADS

# ==============================================================================
# Sequential compute-bound and memory-bound functions
# ==============================================================================

function sumalternatingsignrange(low::Int64, high::Int64)::Int64
# Trivial example of compute-bound work:
# Return sum ((-1)^i)*i for i = low:(high - 1).
    sum::Int64 = 0
    for i = low:high
        sum += (2*((i + 1)%2) - 1)*i  # even numbers positive, odd numbers negative
    end
    return sum
end

function swapinreversedorder(items::Vector{Int64}, low::Int64, high::Int64, n::Int64)::Nothing
# Trivial example of memory-bound work:
# Swap items[low:(low + n - 1)] with items[high:(high - n + 1)].
    startsecs::Float64 = time()
    i = low; j = high; count = 0
    sum = 0
    while count < n
        items[i], items[j] = items[j], items[i]
        i += 1; j -= 1; count += 1
    end
    return
end

# ==============================================================================
# Multitasked compute-bound and memory-bound functions
# ==============================================================================

function timesequentialcomputeboundwork()::Float64
    startsecs::Float64 = time()
    sum = sumalternatingsignrange(0, NUM_COMPUTE_ITERATIONS - 1)
    elapsedsecs = time() - startsecs
    @assert sum == (NUM_COMPUTE_ITERATIONS%2)*(NUM_COMPUTE_ITERATIONS - 1) - NUM_COMPUTE_ITERATIONS÷2
    return elapsedsecs
end

function timemultitaskedcomputeboundwork(numtasks::Int64)::Float64
    startsecs::Float64 = time()
    bounds = Int64.(round.([0:numtasks;]./numtasks*NUM_COMPUTE_ITERATIONS, RoundNearestTiesUp))
    sums = zeros(numtasks)
    Threads.@sync for task = 1:numtasks
        Threads.@spawn sums[task] = sumalternatingsignrange(bounds[task], bounds[task + 1] - 1)
    end
    total = sum(sums)
    elapsedsecs = time() - startsecs
    @assert total == (NUM_COMPUTE_ITERATIONS%2)*(NUM_COMPUTE_ITERATIONS - 1) - NUM_COMPUTE_ITERATIONS÷2
    return elapsedsecs
end

items = [1:NUM_MEMORY_ITEMS;]

function timesequentialmemoryboundwork()::Float64
    global items
    # items = [1:NUM_MEMORY_ITEMS;]
    startsecs::Float64 = time()
    swapinreversedorder(items, 1, NUM_MEMORY_ITEMS, NUM_MEMORY_ITEMS÷2)
    elapsedsecs = time() - startsecs
    # @assert items[1] == NUM_MEMORY_ITEMS && all(items[i - 1] - 1 == items[i] for i in 2:NUM_MEMORY_ITEMS)
    return elapsedsecs
end

function timemultitaskedmemoryboundwork(numtasks::Int64)::Float64
    global items
    # items = [1:NUM_MEMORY_ITEMS;]
    startsecs::Float64 = time()
    halflength = length(items)÷2
    bounds = Int64.(round.([0:numtasks;]./numtasks*halflength, RoundNearestTiesUp)) .+ 1
    Threads.@sync for task = 1:numtasks
        Threads.@spawn swapinreversedorder(items, bounds[task], length(items) - bounds[task] + 1, bounds[task + 1] - bounds[task])
    end
    elapsedsecs = time() - startsecs
    # @assert items[1] == NUM_MEMORY_ITEMS && all(items[i - 1] - 1 == items[i] for i in 2:NUM_MEMORY_ITEMS)
    return elapsedsecs
end

# ==============================================================================
# Plot multi-tasked speedups for compute-bound and memory-bound work
# ==============================================================================

function timespeedups(timesequentialwork, timemultitaskedwork)::Vector{Float64}
    sequentialtimes = sort([timesequentialwork() for trial in 1:NUM_TIMING_TRIALS])
    sequentialtime = median(sequentialtimes)
    println("sequential,$(sequentialtimes[1]),$(sequentialtimes[NUM_TIMING_TRIALS]),$(sequentialtime),1.0")
    speedups = zeros(MAX_TASKS)
    for numtasks in 1:MAX_TASKS
        multitaskedtimes = sort([timemultitaskedwork(numtasks) for trial in 1:NUM_TIMING_TRIALS])
        multitaskedtime = median(multitaskedtimes)
        speedups[numtasks] = sequentialtime/multitaskedtime
        println("$(numtasks),$(multitaskedtimes[1]),$(multitaskedtimes[NUM_TIMING_TRIALS]),$(multitaskedtime),$(speedups[numtasks])")
    end
    return speedups
end

function plotspeedups()::Nothing
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2.5))
    ax.set_xlabel("Number of Tasks")
    ax.set_xlim(0, MAX_TASKS)
    ax.set_xticks([0:NUM_OPERATINGSYSTEM_THREADS:MAX_TASKS;])
    ax.set_ylabel("Speedup Over Sequential")
    ax.set_ylim(0, MAX_SPEEDUP)
    ax.set_yticks([0:3:MAX_SPEEDUP;])
    ax.grid(true)
    ax.plot([0, MAX_SPEEDUP], [0, MAX_SPEEDUP], color="darkgrey", linestyle="dashed")
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)

    println("Compute-bound:")
    computeboundspeedups = timespeedups(timesequentialcomputeboundwork, timemultitaskedcomputeboundwork)
    ax.plot([1:MAX_TASKS;], computeboundspeedups, color="dodgerblue", linestyle="solid", label="Compute-bound")
    println()

    println("Memory-bound:")
    memoryboundspeedups = timespeedups(timesequentialmemoryboundwork, timemultitaskedmemoryboundwork)
    ax.plot([1:MAX_TASKS;], memoryboundspeedups, color="red", linestyle="solid", label="Memory-bound")
    println()

    ax.legend(loc="center right", edgecolor="black", fancybox=false)
    fig.tight_layout(pad=0.25)  # See also: constrained_layout
    display(plt.gcf())
    plt.savefig("PyPlot.MultithreadedPerformance\\SpeedupsVsNumTasksComputeBoundAndMemoryBound.png")
end

# ==============================================================================
# Main program
# ==============================================================================

println()
println("Hello")
plotspeedups()
println("Goodbye")

end
