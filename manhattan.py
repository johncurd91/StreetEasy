import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

streeteasy = pd.read_csv("manhattan.csv")
df = pd.DataFrame(streeteasy)

# Split dataset into dependent and independent variables
x = df[["bedrooms",
        "bathrooms",
        "size_sqft",
        "min_to_subway",
        "floor",
        "building_age_yrs",
        "no_fee",
        "has_roofdeck",
        "has_washer_dryer",
        "has_doorman",
        "has_elevator",
        "has_dishwasher",
        "has_patio",
        "has_gym"]]

y = df[["rent"]]

# Split datasets into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=6)

# Check shape of data
"""
print(f"The shape of the datasets are:\n"
      f"x_train: {x_train.shape}\n"
      f"x_test: {x_test.shape}\n"
      f"y_train: {y_train.shape}\n"
      f"y_test: {y_test.shape}\n")
"""

# Create regression model
mlr = LinearRegression()
mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

# Visualise model
plt.scatter(y_test, y_predict, alpha=0.4)

plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual vs Predicted prices")

plt.show()

# Check coefficients
print("Model coefficients:")
for i in range(len(x.columns)):
    print(f"{x.columns[i]}: {mlr.coef_[0][i]}")
print("")

# Plot relationships between between single variables
"""
for variable in x.columns:
    plt.scatter(df[[variable]], df[["rent"]], alpha=0.4)
    plt.xlabel(variable)
    plt.ylabel("rent")
    plt.title(f"{variable} vs rent")
    plt.show()
"""

# Score model
print(f"Train score: {mlr.score(x_train, y_train)}")
print(f"Test score: {mlr.score(x_test, y_test)}\n")

residuals = y_predict - y_test

plt.scatter(y_predict, residuals, alpha=0.4)
plt.xlabel("y_predict")
plt.ylabel("residuals")
plt.title('Residual Analysis')

plt.show()

# Test prediction using example apartment from https://streeteasy.com/rental/2177438
"""
meeker_ave = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
predict = mlr.predict(meeker_ave)

print(f"Predicted price of 534 Meeker Ave is: ${round(predict[0][0], 2)}.")
"""

