--------------------------------------------------------------------------------
-- BuildTables.sql
--
-- Create Microsoft SQL Server tables for football goal scoring analysis.
--
-- Tables loaded from external sources (data-sourcing and loading code in Scala + JDOM + Squeryl):
--
-- * Datasource: e.g., FootballData.co.uk
-- * Season: e.g., 2001-02
-- * League: e.g., English Premier League
-- * DatasourceLeague: e.g., FootballData.co.uk, English Premier League
-- * SeasonLeague: e.g., 2001-02, English Premier League
-- * Team: e.g., Manchester United
-- * DatasourceTeam: e.g., FootballData.co.uk, Manchester United
-- * Bookmaker: e.g., William Hill
-- * DatasourceBookmaker: e.g., FootballData.co.uk, William Hill
-- * CompetingTeam: e.g., 2001-02, English Premier League, Manchester United
-- * Match: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United
-- * Result: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United, FootballData.co.uk, halftime 2-1, fulltime 4-3
-- * Event: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United, FootballData.co.uk, ordinary goal for Manchester United scored by Ruud van Nistelrooy in the 29th minute
-- * Odds: e.g., 2001-09-15, 2001-02, English Premier League, Newcastle United vs Manchester United, FootballData.co.uk, William Hill, home win odds = 1.9, draw odds = 3.25, away win odds = 3.4
--
-- Tables calculated using Bivariate Poisson model (calculation and loading code in Julia):
--
-- * ProbsImpliedMean: Implied goal-scoring rates for two teams with given win probabilities and goal-scoring correlation
--------------------------------------------------------------------------------

IF object_id('Datasource') IS NULL
CREATE TABLE Datasource (
  DatasourceId BIGINT IDENTITY(1, 1) PRIMARY KEY,
  Keyname VARCHAR(20) NOT NULL UNIQUE,
  CHECK(LEN(Keyname) > 0),
  Description VARCHAR(100) NOT NULL UNIQUE,
  CHECK(LEN(Description) > 0),
  Url VARCHAR(200) NOT NULL,
  CHECK(LEN(Url) > 0),
  HasOdds BIT NOT NULL,
  HasEvents BIT NOT NULL,
  HasSameMinuteOrder BIT NOT NULL,
  CHECK(HasSameMinuteOrder = 0 OR HasEvents IS NOT NULL),
  HasAddedMinutes BIT NOT NULL,
  CHECK(HasAddedMinutes = 0 OR HasEvents IS NOT NULL),
  HasTimeEvents BIT NOT NULL,
  CHECK(HasTimeEvents = 0 OR HasEvents IS NOT NULL),
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Datasource_Keyname')
CREATE UNIQUE INDEX ix_Datasource_Keyname
ON Datasource (Keyname)
GO

--------------------------------------------------------------------------------

IF object_id('Season') IS NULL
CREATE TABLE Season (
  SeasonYear INT PRIMARY KEY,
  CHECK(2000 <= SeasonYear AND SeasonYear <= 2020)
)
GO

--------------------------------------------------------------------------------

IF object_id('League') IS NULL
CREATE TABLE League (
  LeagueId BIGINT IDENTITY(1, 1) PRIMARY KEY,
  Keyname VARCHAR(20) NOT NULL UNIQUE,
  CHECK(LEN(Keyname) > 0),
  Fullname VARCHAR(100) NOT NULL UNIQUE,
  CHECK(LEN(Fullname) > 0),
  Level INT NOT NULL,
  CHECK(1 <= Level AND Level <= 99)
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_League_Keyname')
CREATE UNIQUE INDEX ix_League_Keyname
ON League(Keyname)
GO

--------------------------------------------------------------------------------

IF object_id('DatasourceLeague') IS NULL
CREATE TABLE DatasourceLeague (
  DatasourceId BIGINT,
  FOREIGN KEY (DatasourceId) REFERENCES Datasource(DatasourceId),
  LeagueName VARCHAR(50),
  CHECK(LEN(LeagueName) > 0),
  PRIMARY KEY (DatasourceId, LeagueName),
  LeagueId BIGINT,
  FOREIGN KEY (LeagueId) REFERENCES League(LeagueId)
)
GO

--------------------------------------------------------------------------------

IF object_id('SeasonLeague') IS NULL
CREATE TABLE SeasonLeague (
  SeasonYear INT,
  FOREIGN KEY (SeasonYear) REFERENCES Season(SeasonYear),
  LeagueId BIGINT,
  FOREIGN KEY (LeagueId) REFERENCES League(LeagueId),
  PRIMARY KEY (SeasonYear, LeagueId),
  StartDate DATE NOT NULL,
  CHECK(YEAR(StartDate) = SeasonYear),
  EndDate DATE NOT NULL,
  CHECK(YEAR(EndDate) = SeasonYear + 1),
  CHECK(StartDate < EndDate)
)
GO

--------------------------------------------------------------------------------

IF object_id('Team') IS NULL
CREATE TABLE Team (
  TeamId BIGINT IDENTITY(1, 1) PRIMARY KEY,
  Keyname VARCHAR(20) NOT NULL UNIQUE,
  CHECK(LEN(Keyname) > 0),
  Fullname VARCHAR(100) NOT NULL UNIQUE,
  CHECK(LEN(Fullname) > 0)
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Team_Keyname')
CREATE UNIQUE INDEX ix_Team_Keyname
ON Team (Keyname)
GO

--------------------------------------------------------------------------------

IF object_id('DatasourceTeam') IS NULL
CREATE TABLE DatasourceTeam (
  DatasourceId BIGINT,
  FOREIGN KEY (DatasourceId) REFERENCES Datasource(DatasourceId),
  TeamName VARCHAR(50),
  CHECK(LEN(TeamName) > 0),
  PRIMARY KEY (DatasourceId, TeamName),
  TeamId BIGINT,
  FOREIGN KEY (TeamId) REFERENCES Team(TeamId)
)
GO

--------------------------------------------------------------------------------

IF object_id('Bookmaker') IS NULL
CREATE TABLE Bookmaker (
  BookmakerId BIGINT IDENTITY(1, 1) PRIMARY KEY,
  Keyname VARCHAR(20) NOT NULL UNIQUE,
  CHECK(LEN(Keyname) > 0),
  Fullname VARCHAR(100) NOT NULL UNIQUE,
  CHECK(LEN(Fullname) > 0),
  Url VARCHAR(200) NOT NULL,
  CHECK(LEN(Url) > 0),
  IsConsolidated BIT NOT NULL
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Bookmaker_Keyname')
CREATE UNIQUE INDEX ix_Bookmaker_Keyname
ON Bookmaker (Keyname)
GO

--------------------------------------------------------------------------------

IF object_id('DatasourceBookmaker') IS NULL
CREATE TABLE DatasourceBookmaker (
  DatasourceId BIGINT,
  FOREIGN KEY (DatasourceId) REFERENCES Datasource(DatasourceId),
  BookmakerName VARCHAR(50),
  CHECK(LEN(BookmakerName) > 0),
  PRIMARY KEY (DatasourceId, BookmakerName),
  BookmakerId BIGINT,
  FOREIGN KEY (BookmakerId) REFERENCES Bookmaker(BookmakerId)
)
GO

--------------------------------------------------------------------------------

IF object_id('CompetingTeam') IS NULL
CREATE TABLE CompetingTeam (
  SeasonYear INT,
  LeagueId BIGINT,
  FOREIGN KEY (SeasonYear, LeagueId) REFERENCES SeasonLeague(SeasonYear, LeagueId),
  TeamId BIGINT,
  FOREIGN KEY (TeamId) REFERENCES Team(TeamId),
  PRIMARY KEY (SeasonYear, LeagueId, TeamId),
  UNIQUE (SeasonYear, TeamId)
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_CompetingTeam_SeasonLeague')
CREATE INDEX ix_CompetingTeam_SeasonLeague
ON CompetingTeam (SeasonYear, LeagueId)
GO

--------------------------------------------------------------------------------

IF object_id('Match') IS NULL
CREATE TABLE Match (
  MatchId BIGINT IDENTITY(1, 1) PRIMARY KEY,
  MatchDate DATE NOT NULL,
  CHECK (STR(SeasonYear)+'-08-01' <= MatchDate AND MatchDate <= STR(SeasonYear + 1)+'-07-30'),
  SeasonYear INT,
  LeagueId BIGINT,
  FOREIGN KEY (SeasonYear, LeagueId) REFERENCES SeasonLeague(SeasonYear, LeagueId),
  HomeTeamId BIGINT,
  FOREIGN KEY (SeasonYear, LeagueId, HomeTeamId) REFERENCES CompetingTeam(SeasonYear, LeagueId, TeamId),
  AwayTeamId BIGINT,
  FOREIGN KEY (SeasonYear, LeagueId, AwayTeamId) REFERENCES CompetingTeam(SeasonYear, LeagueId, TeamId),
  CHECK(HomeTeamId <> AwayTeamId),
  UNIQUE (MatchDate, HomeTeamId),
  UNIQUE (MatchDate, AwayTeamId),
  UNIQUE(SeasonYear, HomeTeamId, AwayTeamId) -- Holds for English leagues but not for all leagues
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_All')
CREATE UNIQUE INDEX ix_Match_All
ON Match(MatchDate, SeasonYear, LeagueId, HomeTeamId, AwayTeamId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_DateLeague')
CREATE INDEX ix_Match_DateLeague
ON Match (MatchDate, LeagueId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_DateTeams')
CREATE UNIQUE INDEX ix_Match_DateTeams
ON Match (MatchDate, HomeTeamId, AwayTeamId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_SeasonLeague')
CREATE INDEX ix_Match_SeasonLeague
ON Match (SeasonYear, LeagueId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_SeasonTeams')
CREATE UNIQUE INDEX ix_Match_SeasonTeams
ON Match (SeasonYear, HomeTeamId, AwayTeamId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_SeasonHomeTeam')
CREATE INDEX ix_Match_SeasonHomeTeam
ON Match (SeasonYear, HomeTeamId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Match_SeasonAwayTeam')
CREATE INDEX ix_Match_SeasonAwayTeam
ON Match (SeasonYear, AwayTeamId)
GO

--------------------------------------------------------------------------------

IF object_id('Result') IS NULL
CREATE TABLE Result (
  MatchId BIGINT,
  FOREIGN KEY (MatchId) REFERENCES Match(MatchId),
  DatasourceId BIGINT, 
  FOREIGN KEY (DatasourceId) REFERENCES Datasource(DatasourceId),
  PRIMARY KEY (MatchId, DatasourceId),
  HalftimeHomeScore TINYINT NOT NULL,
  HalftimeAwayScore TINYINT NOT NULL,
  FulltimeHomeScore TINYINT NOT NULL,
  FulltimeAwayScore TINYINT NOT NULL,
  CHECK (0 <= FulltimeHomeScore AND FulltimeHomeScore <= 20),
  CHECK (0 <= FulltimeAwayScore AND FulltimeAwayScore <= 20),
  CHECK (0 <= HalftimeHomeScore AND HalftimeHomeScore <= FulltimeHomeScore),
  CHECK (0 <= HalftimeAwayScore AND HalftimeAwayScore <= FulltimeAwayScore)
)
GO

--------------------------------------------------------------------------------

IF object_id('Event') IS NULL
CREATE TABLE Event (
  MatchId BIGINT,
  FOREIGN KEY (MatchId) REFERENCES Match(MatchId),
  DatasourceId BIGINT, 
  FOREIGN KEY (DatasourceId) REFERENCES Datasource(DatasourceId),
  FOREIGN KEY (MatchId, DatasourceId) REFERENCES Result(MatchId, DatasourceId),
  Minute TINYINT NOT NULL,
  AddedMinute TINYINT NULL, 
  Second TINYINT NULL,
  CHECK(0 <= Minute AND Minute <= 90),
  CHECK(AddedMinute IS NULL OR 0 <= AddedMinute),
  CHECK(AddedMinute IS NULL OR Minute = 45 OR Minute = 90),
  CHECK(Second IS NULL OR (0 <= Second AND Second <= 59)),
  CHECK(Second IS NOT NULL OR (1 <= Minute AND (AddedMinute IS NULL OR 1 <= AddedMinute))),
  CHECK(NOT(Minute = 0 OR (AddedMinute IS NOT NULL AND AddedMinute = 0)) OR Second IS NOT NULL),
  CHECK(NOT(Second IS NOT NULL AND Second = 0) OR (Minute <> 0 AND (AddedMinute IS NOT NULL OR AddedMinute <> 0))),
  CHECK(NOT(Minute = 90 AND AddedMinute IS NULL) OR (Second IS NULL OR Second = 0)),
  CHECK(NOT((Minute = 45 OR Minute = 90) AND AddedMinute = 0 AND Second IS NULL)),
  SameMinuteOrder TINYINT NOT NULL,
  CHECK(1 <= SameMinuteOrder),
  EventType TINYINT NOT NULL,
  CHECK(1 <= EventType AND EventType <= 4), -- 1 = Goal, 2 = RedCard, 3 = Halftime, 4 = Fulltime
  CHECK(EventType <> 3 OR (Minute = 45 AND (AddedMinute IS NOT NULL OR (Second IS NULL OR Second = 0)))),
  CHECK(EventType <> 4 OR ((45 < Minute AND Minute < 90 AND AddedMinute IS NULL) OR (Minute = 90 AND (AddedMinute IS NOT NULL OR (Second IS NULL OR Second = 0))))),
  Side TINYINT NULL,
  CHECK(Side = 1 OR Side = 2), -- 1 = Home, 2 = Away
  CHECK((EventType >= 3 AND Side IS NULL) OR (EventType < 3 AND Side IS NOT NULL)),
  Player nvarchar(50) NULL,
  CHECK(LEN(Player) > 0),
  CHECK((Side IS NULL AND Player IS NULL) OR (Side IS NOT NULL AND Player IS NOT NULL)),
  GoalType TINYINT NULL,
  CHECK((EventType <> 1 AND GoalType IS NULL) OR (EventType = 1 AND GoalType IS NOT NULL)),
  CHECK(GoalType = 1 OR GoalType = 2 OR GoalType = 3) -- NULL = NOT goal, 1 = Normal, 2 = OwnGoal, 3 = Penalty
)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Event_MatchDatasource')
CREATE INDEX ix_Event_MatchDatasource
ON Event (MatchId, DatasourceId)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Event_MatchDatasourceTime')
CREATE INDEX ix_Event_MatchDatasourceTime
ON Event (MatchId, DatasourceId, Minute, AddedMinute, SameMinuteOrder)
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'ix_Event_All')
CREATE INDEX ix_Event_All
ON Event (MatchId, DatasourceId, Minute, AddedMinute, SameMinuteOrder, Second, EventType, Side, Player, GoalType)
GO

--------------------------------------------------------------------------------

IF object_id('Odds') IS NULL
CREATE TABLE Odds (
  MatchId BIGINT,
  FOREIGN KEY (MatchId) REFERENCES Match(MatchId),
  DatasourceId BIGINT, 
  FOREIGN KEY (DatasourceId) REFERENCES Datasource(DatasourceId),
  BookmakerId BIGINT, 
  FOREIGN KEY (BookmakerId) REFERENCES Bookmaker(BookmakerId),
  PRIMARY KEY (MatchId, DatasourceId, BookmakerId),
  AwaywinOdds float NOT NULL,
  DrawOdds float NOT NULL,
  HomewinOdds float NOT NULL,
  CHECK(1.0 <= AwaywinOdds AND AwaywinOdds < 1000.0),
  CHECK(1.0 <= DrawOdds AND DrawOdds < 1000.0),
  CHECK(1.0 <= HomewinOdds AND HomewinOdds < 1000.0)
)
GO

--------------------------------------------------------------------------------

IF object_id('ProbsImpliedMeans') IS NULL
CREATE TABLE ProbsImpliedMeans (
  Correlation DECIMAL(4, 3),
  AwaywinProb3DP DECIMAL(4, 3),
  HomewinProb3DP DECIMAL(4, 3),
  PRIMARY KEY (Correlation, AwaywinProb3DP, HomewinProb3DP),
  CHECK(0.0 <= Correlation AND Correlation < 1.000),
  CHECK(0.0 < AwaywinProb3DP AND AwaywinProb3DP < 1.0),
  CHECK(0.0 < HomewinProb3DP AND HomewinProb3DP < 1.0),
  CHECK(AwaywinProb3DP + HomewinProb3DP < 1.0),
  HomeGoalsMean DECIMAL(6, 3) NOT NULL,
  AwayGoalsMean DECIMAL(6, 3) NOT NULL,
  CHECK(0.0 < HomeGoalsMean AND HomeGoalsMean < 1000.0),
  CHECK(0.0 < AwayGoalsMean AND AwayGoalsMean < 1000.0),
  CHECK((AwaywinProb3DP < HomewinProb3DP AND AwayGoalsMean < HomeGoalsMean) OR (AwaywinProb3DP = HomewinProb3DP AND AwayGoalsMean = HomeGoalsMean) OR (AwaywinProb3DP > HomewinProb3DP AND AwayGoalsMean > HomeGoalsMean))
)
GO

--------------------------------------------------------------------------------