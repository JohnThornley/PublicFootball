# DataSchema

SQL scripts to create Microsoft SQL Server tables and views in the football database.

### Tables loaded from external sources (data-sourcing and loading code in Scala + JDOM + Squeryl)

- Datasource: e.g., FootballData.co.uk
- Season: e.g., 2001-02
- League: e.g., English Premier League
- DatasourceLeague: e.g., FootballData.co.uk, English Premier League
- SeasonLeague: e.g., 2001-02, English Premier League
- Team: e.g., Manchester United
- DatasourceTeam: e.g., FootballData.co.uk, Manchester United
- Bookmaker: e.g., William Hill
- DatasourceBookmaker: e.g., FootballData.co.uk, William Hill
- CompetingTeam: e.g., 2001-02, English Premier League, Manchester United
- Match: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United
- Result: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United, FootballData.co.uk, halftime 2-1, fulltime 4-3
- Event: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United, FootballData.co.uk, ordinary goal for Manchester United scored by Ruud van Nistelrooy in the 29th minute
- Odds: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United, FootballData.co.uk, William Hill, home win odds = 1.9, draw odds = 3.25, away win odds = 3.4

### Tables calculated using Bivariate Poisson model (calculation and loading code in Julia)

- ProbsImpliedMean: Implied goal-scoring rates for two teams with given win probabilities and goal-scoring correlation

### Views

- OddsImpliedProbs: Matches enriched with best odds and odds-implied result probabilities
- MinuteResult: Match states at every minute interval