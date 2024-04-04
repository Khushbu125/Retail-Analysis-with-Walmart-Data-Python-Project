import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:\Khushbu\Data Science\Python Project\Walmart_Store_sales.csv')
print(df.head())

# changing the data type of the ‘Date’ column because it is an object type

from datetime import datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print(df.dtypes)

# Statistical Tasks-
# Which store has maximum sales?
total_sales = df.groupby('Store')['Weekly_Sales'].sum().round().sort_values(ascending=False)
print(pd.DataFrame(total_sales).head(1))

# Which store has maximum standard deviation i.e., the sales vary a lot.
# Also, find out the coefficient of mean to standard deviation
df_std = df.groupby('Store')['Weekly_Sales'].std().round().sort_values(ascending=False)
print(pd.DataFrame(df_std).head())

# Store which has maximum Standard Deviation
print(pd.DataFrame(df_std).head(1))

# Coefficient of mean to standard deviation
store14 = df[df.Store == 14].Weekly_Sales
mean_to_stddev = store14.std()/store14.mean()*100
print(mean_to_stddev, '%')

# Which store/s has a good quarterly growth rate in Q3’2012?
# Finding the Q2 sales then Q3 sales,then taking out the difference to get the growth rate.
q2_sales = df[(df['Date'] >= '2012-04-01') & (df['Date'] <= '2012-06-30')].groupby('Store')['Weekly_Sales'].sum().round()
q3_sales = df[(df['Date'] >= '2012-07-01') & (df['Date'] <= '2012-09-30')].groupby('Store')['Weekly_Sales'].sum().round()
# Growth rate = ((present-past)/past)*100
df_2012 = pd.DataFrame({'Q2 Sales': q2_sales, 'Q3 Sales': q3_sales, 'Difference': (q3_sales-q2_sales), 'Growth Rate %': (q3_sales-q2_sales)/q2_sales*100}).sort_values(by='Growth Rate %', ascending=False).head()
print(df_2012)

max_sales_2012Q3 = df_2012.groupby('Store')['Growth Rate %'].sum()
print(f'max_sales_2012 Q3 = ', max_sales_2012Q3.idxmax())

# Some holidays have a negative impact on sales.Find out holidays that have higher
# sales than the mean sales in the non-holiday season for all stores together.
# We have 4 Holiday Events, (1) Super Bowl: 12-Feb-10, 11-Feb-11, 10-Feb-12, 8-Feb-13,
# (2) Labour Day: 10-Sep-10,9-Sep-11, 7-Sep-12, 6-Sep-13,
# (3) Thanksgiving: 26-Nov-10, 25-Nov-11, 23-Nov-12, 29-Nov-13,
# (4) Christmas: 31-Dec-10, 30-Dec-11, 28-Dec-12, 27-Dec-13.

# Calculating the holiday event sales of each of the events and then find the non-holiday sales.
# Holiday events
Super_Bowl = ['12-02-2010', '11-02-2011', '10-02-2012', '08-02-2013']
Labour_Day = ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06']
Thanksgiving = ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29']
Christmas = ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']

# Convert holiday lists to datetime objects if they are not already datetime objects
Super_Bowl = pd.to_datetime(Super_Bowl)
Labour_Day = pd.to_datetime(Labour_Day)
Thanksgiving = pd.to_datetime(Thanksgiving)
Christmas = pd.to_datetime(Christmas)

# Calculate sales for each holiday
Super_Bowl_Sales = round(df[df.Date.isin(Super_Bowl)]['Weekly_Sales'].mean(), 2)
Labour_Day_Sales = round(df[df.Date.isin(Labour_Day)]['Weekly_Sales'].mean(), 2)
Thanksgiving_Sales = round(df[df.Date.isin(Thanksgiving)]['Weekly_Sales'].mean(), 2)
Christmas_Sales = round(df[df.Date.isin(Christmas)]['Weekly_Sales'].mean(), 2)

# Non-holiday Sales and Comparison
non_holiday_sales = round(df[df['Holiday_Flag'] == 0]['Weekly_Sales'].mean(), 2)
print(non_holiday_sales)

print(pd.DataFrame([{'Super Bowl Sales': Super_Bowl_Sales,
               'Labour day Sales': Labour_Day_Sales,
               'Thanksgiving Sales': Thanksgiving_Sales,
               'Christmas Sales': Christmas_Sales,
               'non holiday Sales': non_holiday_sales}]))

# Thanksgiving has the highest sales (1,471,273.43) than non-holiday sales (1,041,256.38)

# Provide a monthly and semester view of sales in units and give insights.
# Plotting a month-wise bar graph for weekly sales to get an idea about which month has the maximum sales,
# then will plot the semester-wise bar graph for weekly sales to get some insights about the semester's weekly sales.

df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day

# Monthwise Sales
plt.figure(figsize=(15,7), dpi=85)
plt.bar(df['month'],df['Weekly_Sales'])
plt.xlabel('Months')
plt.ylabel('Weekly Sales')
plt.title('Monthwise Sales')
plt.show()

# Yearly Sales
plt.figure(figsize=(15, 7), dpi=85)
df.groupby('year')[['Weekly_Sales']].sum().plot(kind='bar',legend=False)
plt.title('Yearly Sales')

# Semesterwise Sales
df['semester'] = np.where(df['month']< 7, 1, 2)
df.head()
plt.figure(figsize=(15, 8))
semester = sns.barplot(x='semester', y='Weekly_Sales', data=df)
plt.show()

# Insights drawn-
# (1)December month has the highest weekly sales.
# (2) Semester 2 has the highest weekly sales.

# Model Building - First, define dependent and independent variables.
# Here, store, fuel price, CPI, unemployment, day, month,
# and year are the independent variables and weekly sales is the dependent variable.
# Now, it’s time to train the model.
# Import train_test_spit from sklearn.model_selection and train 80% of the data and test on the rest 20% of the data.

# Define independent and dependent variable
# Select features and target
x = df[['Store', 'Fuel_Price', 'CPI', 'Unemployment', 'day', 'month', 'year']]
y = df['Weekly_Sales']

from sklearn.model_selection import train_test_split
# Split data to train and test (0.80:0.20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
# Linear Regression model
print('Linear Regression:')
print()
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print('Accuracy:', reg.score(x_train, y_train)*100)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.scatterplot(x=y_pred, y=y_test)
plt.show()

# Random Forest Regressor
print('Random Forest Regressor:')
rfr = RandomForestRegressor(n_estimators=400, max_depth=15, n_jobs=5)
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_test)
print('Accuracy:', rfr.score(x_test, y_test)*100)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.scatterplot(x=y_pred, y=y_test)
plt.show()
# Here, we have used 2 different algorithms to know which model to use to predict the weekly sales.
# Linear Regression is not an appropriate model to use as accuracy is very low.
# However, Random Forest Regression gives an accuracy of almost 89%. so, it is the best model to forecast weekly sales.

# Change dates into days by creating new variable.

df['day'] = pd.to_datetime(df['Date']).dt.day_name()
df.head()

experiment_day_start = 5
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['exp_day'] = (df['Date']-df['Date'].min()).dt.days + experiment_day_start
df.head()

from sklearn.linear_model import LinearRegression
from scipy import stats
# Weekly sales vs Unemployment
x = df['Unemployment']
y = df['Weekly_Sales']
plt.scatter(x, y)
plt.show()
slope,intercept, r, p, std_err = stats.linregress(x, y)
print(r)            # r should be between -1 to 1


def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

# Weekly_Sales vs exp_day
x = df['exp_day']
y = df['Weekly_Sales']
plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)                # r should be between -1 to 1



def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


# Weekly sales vs CPI
x = df['CPI']
y = df['Weekly_Sales']
plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err = stats.linregress(x, y)
print(r)            # r should be between -1 to 1



def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

# Weekly sales vs Fuel price
x = df['Fuel_Price']
y = df['Weekly_Sales']
plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err=stats.linregress(x, y)
print(r)                # r should be between -1 to 1


def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

# Weekly sales vs Holidays
x = df['Holiday_Flag']
y = df['Weekly_Sales']
plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err=stats.linregress(x, y)
print(r)                # r should be between -1 to 1


def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()

# Weekly sales vs Temperature
x = df['Temperature']
y = df['Weekly_Sales']
plt.scatter(x, y)
plt.show()
slope, intercept, r, p, std_err=stats.linregress(x, y)
print(r)      # r should be between -1 to 1



def myfunc(x):
  return slope * x + intercept


mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


df.to_csv('D:\Khushbu\Data Science\Python Project\Walmart_Store_sales_new.csv')
