# Project Proposal
#### Lila Sargent, Atticus Tarleton, Ian Sargent, Grace Teller
[Data Source](https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-2/jsu2-fbtj/about_data)

## Data Overview
1. Total ridership aggregated by Month x Day_of_Week x Hour x Origin_Station (~853,000 rows)
2. Total Ridership aggregated by Origin_Station x Destination_Station Combinations (~116 million rows)

NOTE: Calculate a distance variable between origin and destination stations using coordinates

## Metadata
This dataset was retrieved from the New York Open Data Program, an open-source repository for New York State’s publicly available datasets. It contains Metropolitan Transportation Authority (MTA) ridership data estimated and collected from MetroCard swipe and contactless MTA payment systems (OMNY). The original dataset represents aggregated ridership from each origin-destination combination along with the month, hour, and day of the week.

#### Variables:
- Timestamp (Year, Month, Hour, Day of Week)  |  Datetime
- Origin / Destination Station ID + Name  |  Number/Text
- Station Latitude / Longitude Coordinates  |  Number
- Estimated Average Ridership  |  Number
- Origin Point (Geospatial)  |  Point Feature

## Research Questions
1.	Can we accurately forecast station ridership/demand?
2.	Which origin stations are the hardest to predict ridership?
3.	Do seasons contribute to ridership patterns?
4.	Which seasons have the greatest positive and negative deviation from the yearly average?
5.	Which stations have different demand patterns?
6.	Which pair of stations has the largest passenger flow?
7.	Which stations have the largest inbound/outbound traffic?
8.	Do subway stations cluster into distinct communities?

## Methods
- K-Means Clustering or DBSCAN Origin/Destination Stations
- Predicting Total Ridership (Ensemble, Linear Regression, Lasso Feature Selection)
- Time Series Forecasting (ARIMA, SARIMA)
- Principal Component Analysis to Identify Distinct Station Categories

## Graphs
- Line graph of estimated total ridership by hour
- Additionally, could separate into multiple line plots by day of the week
- Furthermore, could choose a specific origin or destination point
- Bar graph of distribution of ridership for all destinations from a given origin point, or vice versa
- Stacked histogram for average estimated daily ridership by origin or destination station
- Line graph of total estimated daily ridership for the whole year or individual month
- Can also probably label some of the spikes in ridership with specific events such as New York Marathon
