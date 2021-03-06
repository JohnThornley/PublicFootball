# Julia

Julia code for bivariate Poisson model and data analysis and the resulting plots.
Also, a script and plot from some initial experimentation (not related to football analysis) with multithreaded performance speedups.
The crux of the code is **Lib\\BivariatePoisson.jl**.

### Folders:

- **Lib** - bivariate Poisson model and related utilities

- **Script.DataCalcs** - scripts for data calculation and persistence

- **Script.PyPlot.BivariatePoissonPaper** - scripts for creating analysis and plots (via PyPlot) used in bivariate Poisson paper
- **PyPlot.BivariatePoissonPaper** - plots produced by scripts in **Script.PyPlot.BivariatePoissonPaper**

- **Script.Plots.NotPaper** - scripts for creating analysis and plots (via Plots) not used in bivariate Poisson paper
- **Plots.NotPaper** -- plots produced by scripts in **Script.Plots.BivariatePoissonPaper**

- **Script.PyPlot.MultithreadedPerformance** - scripts for experimenting with multithreaded execution speedups
- **PyPlot.MultithreadedPerformance** - plots produced by scripts in **Script.PyPlot.MultithreadedPerformance**

### Runnability:

The following scripts are runnable without the database:

- Lib\\BivariatePoissonTests.jl (lots of unit tests)
- Lib\\UtilTests.jl

- Script.PyPlot.BivariatePoissonPaper\\PyPlotProbresultsFromMeans.jl
- Script.PyPlot.BivariatePoissonPaper\\PyPlotMeansFromProbresults.jl (lots of computation)

- Script.Plots.NotPaper\\PlotProbscoreFromLambdas.jl
- Script.Plots.NotPaper\\PlotProbscoreFromMeans.jl
- Script.Plots.NotPaper\\PlotProbresultsFromLambdas.jl
- Script.Plots.NotPaper\\PlotLambdasFromProbresults.jl (lots of computation)

- Script.PyPlot.MultithreadedPerformance\\PlotSpeedupsVsNumTasksComputeBoundAndMemoryBound.jl
