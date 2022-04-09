# PublicFootball

Julia code and SQL data schema related to football goal-scoring analysis - for interested parties to browse.
Pulled out of a long-term multi-language (SQL, Scala, R, Julia, LaTeX) personal project in order to
present the Julia code in a focused and hopefully readable context.

### Suggested browsing order:

1. bivariatepoissonpaper.pdf - early draft paper (not ready for publication) describing the data and analysis being performed

2. DataSchema - folder containing the SQL schema for the football data (but not the actual data)

3. Julia - folder containing Julia code and plots for the bivariate Poission model and data analysis/plotting

### Note:

This is not a complete runnable project - which would require having the SQL Server database installed,
the sourced football result data files, and the Scala/Squerl ORM and loader code.
However, scripts and plots that don't depend on the football data are runnable,
e.g., bivariate Poission vs correlatation and inverse bivariate Poisson vs correlation plots.
