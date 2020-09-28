from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import metrics

# Load ADAS-Cog plus data from /data/ folder
base_path = Path(__file__).parent
file_path = (base_path / "./data/data.csv").resolve()

data = pd.read_csv(file_path, header=0)

# Code to plot each variable vs glob_i to observe linearity
#
# original_headers = list(data.columns.values)
# df = pd.DataFrame(data, columns=original_headers)
# print(df.columns.values)

# plt.scatter(df['ADAS_Cog_Total_Score'], df['glob_i'])
# plt.scatter(df['Trail_Making_Test_Score'], df['glob_i'])
# plt.scatter(df['Digit_Span_Score'], df['glob_i'])
# plt.scatter(df['Animal_Total'], df['glob_i'])
# plt.scatter(df['Vegetable_Total'], df['glob_i'])
# plt.scatter(df['DSST'], df['glob_i'])
# plt.show()


# Code to check colinearity of variables
#
# pd.set_option('display.max_columns', 8)
# corrMatrix = data.corr()
# print(corrMatrix)

X = data.iloc[:, [1, 2, 4, 5, 6]].values # excludes Digit Span Score due to non-linear fit
y = data.iloc[:, -2].values

# 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# print('Intercept: \n', regressor.intercept_)
# print('Coefficients: \n', regressor.coef_)

y_pred = regressor.predict(X_test)
# df = pd.DataFrame({'Real values': y_test, 'Predicted values': y_pred})
# pd.set_option('display.max_rows', 100)
# print(df)

print('MAE:', end='')
print(metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', end='')
print(metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', end='')
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', end='')
print(metrics.r2_score(y_test, y_pred))


# 10-fold cross validation
kf = KFold(n_splits=10, shuffle=False)

model = LinearRegression()
scores_r2 = cross_val_score(model, X, y, scoring='r2', cv=kf)
scores_mse = list(map(abs, cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)))
scores_rmse = list(map(np.sqrt, scores_mse))

print('Cross Val R2:')
print(scores_r2)

print('Cross Val MSE:')
print(scores_mse)

print('Cross Val RMSE:')
print(scores_rmse)