import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Load data from csv into pandas dataframe
streeteasy = pd.read_csv("streeteasy.csv")
df = pd.DataFrame(streeteasy)

# Dependent variables
x_value_1 = "size_sqft"
x_value_2 = "building_age_yrs"

# Slice dataframe and fit model
x = df[[x_value_1, x_value_2]]
y = df[["rent"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=6)

ols = LinearRegression()

ols.fit(x_train, y_train)

# Create figure
fig = plt.figure(1, figsize=(11, 9))
plt.clf()

elev = 43.5
azim = -110

ax = Axes3D(fig, elev=elev, azim=azim)

ax.scatter(x_train[[x_value_1]],
           x_train[[x_value_2]],
           y_train, c="k", marker="+")

ax.plot_surface(np.array([[0, 0], [4500, 4500]]),
                np.array([[0, 140], [0, 140]]),
                ols.predict(np.array([[0, 0, 4500, 4500], [0, 140, 0, 140]]).T).reshape((2, 2)), alpha=.7)

ax.set_xlabel("Size (ft$^2$)")
ax.set_ylabel("Building Age (Years)")
ax.set_zlabel("Rent ($)")

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# Display plot
plt.show()
