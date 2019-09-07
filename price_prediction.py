'''
LINEAR, RIDGE AND LASSO REGRESSION
'''
# importing requuired libraries
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import datetime

'''
# DATA Training
'''
df_main = pd.read_csv("AAPL.csv")
print(df_main.head())
df_main['HL_PCT'] = (df_main['High'] - df_main['Low']) / df_main['Adj Close'] * 100.0
df_main['PCT_change'] = (df_main['Adj Close'] - df_main['Open']) / df_main['Open'] * 100.0
df = df_main[['Date', 'Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
print(df.head())
forecast_col = 'Adj Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.03 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
x = np.array(df.drop(['label', 'Date'], 1))
y = np.array(df['label'])
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]

# splitting into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

'''
# Training Different Models 
# 1. Training Linear Regression Model
'''
lreg = LinearRegression()
lreg.fit(x_train, y_train)
lreg_pred = lreg.predict(x_test)

# calculating mse
lreg_mse = np.mean((lreg_pred - y_test)**2)
print('\nMean Sqaured Error = ', lreg_mse)
# calculating coefficients
lreg_coeff = DataFrame(x_train)
lreg_coeff['Coefficient Estimate'] = Series(lreg.coef_)
lreg_score = lreg.score(x_test, y_test)
print('\n\n Linear Regrssion Model performance on Test data = ')
print(lreg_score)
future_set_linear = lreg.predict(x_lately)

'''
#2 Training Ridge Regression Model
'''
ridge = Ridge()
ridge.fit(x_train, y_train)
ridge_pred = ridge.predict(x_test)
future_set_ridge = ridge.predict(x_lately)
ridge_mse = np.mean((ridge_pred-y_test)**2)
print('\n\nMean Squared Error = ', ridge_mse)
# calculating coefficients
ridge_coeff = DataFrame(x_train)
ridge_coeff['Coefficient Estimate'] = Series(ridge.coef_)
ridge_score = ridge.score(x_test, y_test)
print('\n\n Ridge Regression Model performance on Test data = ')
print(ridge_score)


'''
#3 Training Lasso Regression Model
'''

lasso = Lasso()
lasso.fit(x_train, y_train)
lasso_pred = lasso.predict(x_test)
future_set_lasso = lasso.predict(x_lately)

lasso_mse = np.mean((lasso_pred-y_test)**2)
print('\n\nMean Squared Error = ', lasso_mse)
# calculating coefficients
lasso_coeff = DataFrame(x_train)
lasso_coeff['Coefficient Estimate'] = Series(lasso.coef_)
lasso_score = lasso.score(x_test, y_test)
print(format('\n\nLasso Regression Model performance on Test data = '))
print(lasso_score)
'''
Data Processing For Visualization
'''
df_main['Ridge'] = np.nan
# Date form Which Future Prediction Starts
last_date = max(pd.to_datetime(df_main['Date']))
print(last_date)
last_unix = last_date
next_unix = last_unix+datetime.timedelta(1)
print(last_unix)
# Future prediction with Ridge and data mapping and cleaning
for i in future_set_ridge:
    next_date = next_unix
    next_unix = next_unix+datetime.timedelta(1)
    df_main.loc[next_date] = [np.nan for _ in range(len(df_main.columns)-1)]+[i]


df_main['Forecast_Date'] = pd.to_datetime(df_main.index,errors='coerce')
predicted_date = list(df_main.ix[df_main['Date'].astype(str) == 'nan']['Forecast_Date'].astype(str))
df_main['Date'] = df_main['Date'].astype(str).replace('nan', df_main['Forecast_Date'].astype(str))
# History date reduced to last 30 days history only to visualize
plotting_date = str(max(pd.to_datetime(df_main['Date']))-datetime.timedelta(30))
df_main = df_main[df_main['Date'] >= plotting_date]
df_linear = pd.DataFrame({'Date': predicted_date, 'Linear': future_set_linear})
df_lasso = pd.DataFrame({'Date': predicted_date, 'Lasso': future_set_lasso})
df_main = pd.merge(df_main, df_linear, on='Date', how='left')
df_main = pd.merge(df_main, df_lasso, on='Date', how='left')
'''
# Plotting all types of predictions with regression type
'''
df_main['Adj Close'].plot()
df_main['Ridge'].plot()
df_main['Linear'].plot()
df_main['Lasso'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
x = df_main['Date'].astype(str).tolist()
print(x)
ind = np.arange(len(x))  # the x locations for the groups
width = 0.15
plt.xticks(ind, x, fontsize=5, rotation=45)
plt.ylabel('Stock Price')
plt.show()
plt.savefig("prediction_graph.png")
plt.close()

model_score = pd.DataFrame({'Model': ['Linear', 'Ridge', 'Lasso'], 'Score': [lreg_score, ridge_score, lasso_score]})
print(model_score)
