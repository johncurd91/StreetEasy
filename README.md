# StreetEasy
Codecademy Data Science course portfolio project.

The aim of this project was to use data provided on rental properties in New York and produce a model that could predict how much the rent should be for any given property.

## Instructions
Running _main.py_ will use LinearRegression() and train_test_split() from sklearn to produce an OLS model which describes how rent prices are influenced _size_sqft_ and _building_age_yrs_. These variables can be changed near the top by changing _x_value1_ and _X_value2_.

It will also display this data in 3D.

Similarly, running _mlr.py_ will use sklearn to produce a model which describes how  rent prices are influenced by all other variables in _manhattan.csv_ (this can be changed to _brooklyn.csv_ or _queens.csv_ near the top). It will then use this model to predict the rent for a chosen property from StreetEasy.com. Additional functions also exist for visualising the model, performing residual analysis, scoring the model, and examining the coefficients.