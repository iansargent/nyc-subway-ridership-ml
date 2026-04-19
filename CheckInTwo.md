### Decision Tree

#### Data Pre-processing

The majority of our data is qualitative. In order to make up for that, before we
could use a decision tree, we had to One-Hot-Encode three columns, "day_of_the_week", 
"origin_station_complex_name", and "origin_point". This ended up giving us 433 columns in total.

#### Training

After processing the data, we split it into a training and testing set and fit it to an unpruned tree.
This gave us an R-Squared on the testing set of 0.99 and 1.00 on the training set. In other words, it
completely overfit the data.

#### Pruning

Since the tree was overfitting the data, we attempted to prune it. We plotted the accuracy of the decision
tree model in terms of its depth and determined the best depth using a grid search (73). Unsurprisingly,
this still gave us a testing R-Squared of 0.99

#### Feature Importance

Finally, we looked into the feature importances of all 433 columns. Hour of the day ended up being the most
important feature by far, followed by latitude and longitude (and whether or not the origin station was
Times Square). This makes sense, as the combination of latitude and longitude is essentially the same
thing as the origin station itself.

Our next step is to use this information to cut down the number of features given to the decision tree 
to see if this will prevent the tree from overfitting.