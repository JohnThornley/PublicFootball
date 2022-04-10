--------------------------------------------------------------------------------
-- CreateViews.sql
--
-- Create Microsoft SQL Server views for football goal scoring analysis.
--
-- * OddsImpliedProbs: Matches enriched with best odds and odds-implied result probabilities
-- * MinuteResult: Match states at every minute interval
--------------------------------------------------------------------------------

CREATE VIEW OddsImpliedProbs AS
SELECT
  Match.MatchId,
  Match.MatchDate,
  Match.SeasonYear,
  League.LeagueId,
  League.Keyname AS LeagueName,
  HomeTeam.TeamId AS HomeTeamId,
  HomeTeam.Keyname AS HomeTeamName,
  AwayTeam.TeamId AS AwayTeamId,
  AwayTeam.Keyname AS AwayTeamName,
  Odds.AwaywinOdds,
  Odds.DrawOdds,
  Odds.HomewinOdds,
  1/Odds.AwaywinOdds + 1/Odds.DrawOdds + 1/Odds.HomewinOdds AS Overround,
  Odds.AwaywinProb,
  Odds.HomewinProb,
  CONVERT(DECIMAL(4,3), Odds.AwaywinProb) AS AwaywinProb3DP,
  CONVERT(DECIMAL(4,3), Odds.HomewinProb) AS HomewinProb3DP,
  Result.HalftimeHomeScore,
  Result.HalftimeAwayScore,
  Result.FulltimeHomeScore,
  Result.FulltimeAwayScore
FROM
  (SELECT
    Odds.DatasourceId,
    Odds.MatchId,
    AwaywinOdds,
    DrawOdds,
    HomewinOdds,
    1/(AwaywinOdds*(1/AwaywinOdds + 1/DrawOdds + 1/HomewinOdds)) AS AwaywinProb,
    1/(HomewinOdds*(1/AwaywinOdds + 1/DrawOdds + 1/HomewinOdds)) AS HomewinProb
  FROM
    Odds
  WHERE
    Odds.DatasourceId = (SELECT DatasourceId FROM Datasource WHERE Keyname = 'FootballData') AND
    Odds.BookmakerId = (SELECT BookmakerId FROM Bookmaker WHERE Keyname = 'InternalBest')) AS Odds
LEFT JOIN Match ON Match.MatchId = Odds.MatchId
LEFT JOIN League ON League.LeagueId = Match.LeagueId
LEFT JOIN Team AS HomeTeam ON HomeTeam.TeamId = Match.HomeTeamId
LEFT JOIN Team AS AwayTeam ON AwayTeam.TeamId = Match.AwayTeamId 
LEFT JOIN Result ON Result.MatchId = Odds.MatchId AND Result.DatasourceId = Odds.DatasourceId
WHERE Match.MatchDate <= '2020-03-10'  -- 2019-20 season matches after this date played in empty stadiums because of the Covid-19 pandemic.
GO

CREATE VIEW Minute AS
SELECT * FROM (VALUES
         (1),  (2),  (3),  (4),  (5),  (6),  (7),  (8),  (9),
  (10), (11), (12), (13), (14), (15), (16), (17), (18), (19),
  (20), (21), (22), (23), (24), (25), (26), (27), (28), (29),
  (30), (31), (32), (33), (34), (35), (36), (37), (38), (39),
  (40), (41), (42), (43), (44), (45), (46), (47), (48), (49),
  (50), (51), (52), (53), (54), (55), (56), (57), (58), (59),
  (60), (61), (62), (63), (64), (65), (66), (67), (68), (69),
  (70), (71), (72), (73), (74), (75), (76), (77), (78), (79),
  (80), (81), (82), (83), (84), (85), (86), (87), (88), (89),
  (90))
AS Minute(Minute)
GO

CREATE VIEW MinuteResult AS
SELECT
  OddsImpliedProbs.MatchId,
  OddsImpliedProbs.MatchDate,
  OddsImpliedProbs.SeasonYear,
  OddsImpliedProbs.LeagueId,
  OddsImpliedProbs.LeagueName,
  OddsImpliedProbs.HomeTeamId,
  OddsImpliedProbs.HomeTeamName,
  OddsImpliedProbs.AwayTeamId,
  OddsImpliedProbs.AwayTeamName,
  CAST(ProbsImpliedMeans.HomeGoalsMean AS FLOAT) AS PredictedHomeGoalRate,
  CAST(ProbsImpliedMeans.AwayGoalsMean AS FLOAT) AS PredictedAwayGoalRate,
  OddsImpliedProbs.FulltimeHomeScore,
  OddsImpliedProbs.FulltimeAwayScore,
   -- EventType: 1 = Goal, 2 = RedCard, 3 = Halftime, 4 = Fulltime
   -- Side: 1 = Home, 2 = Away
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 1) AS FinalHomeGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 2) AS FinalAwayGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 1) AS FinalHomeRedCards,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 2) AS FinalAwayRedCards,
  Minute.Minute,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 1 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) < Minute.Minute) AS PreHomeGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 2 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) < Minute.Minute) AS PreAwayGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 1 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) < Minute.Minute) AS PreHomeRedCards,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 2 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) < Minute.Minute) AS PreAwayRedCards,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 1 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) = Minute.Minute) AS MinuteHomeGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 2 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) = Minute.Minute) AS MinuteAwayGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 1 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) = Minute.Minute) AS MinuteHomeRedCards,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 2 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) = Minute.Minute) AS MinuteAwayRedCards,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 1 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) > Minute.Minute) AS PostHomeGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 1 and E.Side = 2 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) > Minute.Minute) AS PostAwayGoals,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 1 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) > Minute.Minute) AS PostHomeRedCards,
  (select Count(1) from Event as E where E.MatchId = OddsImpliedProbs.MatchId and E.DatasourceId = Soccerbase.DatasourceId and E.EventType = 2 and E.Side = 2 and dbo.MinuteWithoutSeconds(E.Minute, E.AddedMinute, E.Second) > Minute.Minute) AS PostAwayRedCards
FROM
	OddsImpliedProbs, ProbsImpliedMeans, Minute, Datasource AS Soccerbase
WHERE
	ProbsImpliedMeans.Correlation = 0.13 AND OddsImpliedProbs.AwaywinProb3DP = ProbsImpliedMeans.AwaywinProb3DP AND OddsImpliedProbs.HomewinProb3DP = ProbsImpliedMeans.HomewinProb3DP AND
	Soccerbase.Keyname = 'Soccerbase'
GO

-------------------------------------------------------