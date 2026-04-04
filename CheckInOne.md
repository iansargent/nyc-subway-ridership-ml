
### Data Cleaning
The first thing we did for this project was clean the data.
In terms of this, we have two datasets, origin_data and origin_destination.
More info on what these datasets entail can be found in the README

#### Origin_data
The data cleaning of origin_data was fairly straight forward.
All that needed to be done was to organize the rows, standardize the column names--to remove spaces and capitlaization--
and transform the ridership column to numeric data for better and easier analysis.

#### origin_destination
For origin_destination, we also had to standardize column names and transform the ridership column to numeric data.
Similarly, however, we turned the 'origin_point' and 'destination_point' variables into geometry objects
and added them to a GeoDataFrame so that they can be effectively graphed.
In this GeoDataFrame, we also added the log(ridership), the distance between each station, and a new
ridership per kilometer variable.

### Explporatory Analysis
Now that we have cleaned the data, it's time to look at what the datasets actually contain.

#### Distributions
To start, we focused on visualizing different ridership distributions.
This includes per day of the week, month of the year, hour of the day,
hour of the day for each specific day of the week, season of the year, etc.

These graphs give us a very good idea of the distributions of ridership,
which will be the biggest variable of interest, as well as its trends. For
the next couple of weeks, we are also planning on doing more visualizations
of ridership in terms of origin and destination station.

#### Clustering
To go along with our visualizations, we have also started to look into clustering
the data. The first step of this was feature engineering, where we created a new DataFrame.
For this DataFrame, each row is a different station. From there, each column is the station's
average ridership per hour of the day, day of the week, and month of the year.

After creating this DataFrame, we reset the index and standardized the numbers
to increase efficiency. Specifically, the efficiency of the PCA that transformed the
data into two components. We then used these two components to plot a
representation of the ridership for each station. Unfortunately, there does not appear
to be an obvious pattern within the data, besides a couple possible outliers
such as Times Square.