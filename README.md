# StreetEasy
Codecademy Data Science portfolio project.

The purpose of this project was to use machine learning approaches to analyse data provided on rental properties in New York, available on StreetEasy.com.

## Instructions
Running main.py will use multiple linear regression to fit a model which describes the relationship between two variables (by default: size_sqft and building_age_yrs - feel free to change) and rent. It will then provide a 3D plot to visualise this relationship.

Running any one of manhattan.py, brooklyn.py, queens.py will perform a multiple linear regression to fit a model which describes the relationship between all variables and rent. It will also provide model coefficients for each variable, a visualisation of the model, score the model (training vs test scores), peform residual analysis, and in the case of manhattan.py it will also provide a test prediction on a chosen property.