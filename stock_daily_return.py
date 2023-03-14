
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("UBER.csv")# read dataframe
data = pd.read_csv("AAPL.csv")
print(df.head()) #print first 5 lines of data 
print(df.head(n = 10)) # n=10 represent a number of first lignes to print
print(type(df)) # type of data 
print(df.shape) # number of column and row 
print(df.tail()) #print the last five lines
print(df.tail(n=10)) # n=10 represent a number of last lignes to print 
print(df.describe(include="all"))
print(df.info())
print(df.dtypes)
df["Date"] = pd.to_datetime(df["Date"]) # date is considerte us a objective we need to convertir to date
print(df.info())
print(df.describe(include="all"))
print(df.value_counts()) #count number of value existe
print(df.head())
df.loc[:, "Close"].plot() #print a prol of changing of column close in function of date
plt.title("Close price of the day")
plt.show()
print(df["High"].std())
print(df["Low"].mean())
corr = df.corr()
print(corr)
regr = LinearRegression()
y = df["Date"]
x = df[["Low" , "High"]]
print(regr.fit(x,y))
print(regr.coef_)
print(regr.intercept_)
df["10"] = df["Close"].rolling(10).mean() #moving average of short periode is the most associated with recent change of stock price called fast signal
df["50"] = df["Close"].rolling(50).mean() #moving average with long period reflect the prices changes over long term history called slow signal
print(df.head())
df = df.dropna()
df["Close"].plot(legend=True)
df["10"].plot(legend=True)
df["50"].plot(legend=True)
print(df.head())
plt.show()
#strategie of buying when 10>50 we will buy 
df["Shares"] = [1 if df.loc[ei, "10"]>df.loc[ei, "50"] else 0 for ei in df.index] #create variable send 1 if it's good to buy while 
#the fast signal upper then slow signal 
print(df.head()) 
#daily profite
df["Close1"] = df["Close"].shift(-1) # return a value of close price of day befor 
df["profit"] = [df.loc[ei , "Close1"] - df.loc[ei, "Close"] if df.loc[ei, "Shares"]== 1 else 0 for ei in df.index ] 
#daily profit
#if share=1 the daily profit = close1- close if negative we lose money if positive we win 
# if shares =0 we dont have stock at hand 
print(df.head())
df["profit"].plot()
plt.axhline(y=0, color='red')
plt.title("explain win or lose ") # title name of plot 
plt.show()
df["wealth"] = df["profit"].cumsum() #cumulative wealth
print(df.tail())
df["wealth"].plot()
plt.title("Total money you win is {}".format(df.loc[df.index[-2], "wealth"]))
plt.show()
print("total money you win is ", df.loc[df.index[-2], "wealth"])
print("total money you spend is ", df.loc[df.index[0], "Close"])
#daily return 
df["LogReturn"] = np.log(df["Close"]).shift(-1) - np.log(df["Close"])
df["LogReturn"].hist(bins = 50)
plt.show()
print(data.head()) #print first 5 line of data 
print(data.head(n = 10)) #print first 10  line of data here we have presize a number of lines
print(type(data)) # type of data 
print(data.shape) # number of column and row 
print(data.tail()) #print the last five lines 
print(data.dtypes) # date is considerte us a objective we need to convertir to date
data["Date"] = pd.to_datetime(data["Date"])
print(data.head())
print(data.dtypes)
print(data.value_counts())
data.loc[:, "Close"].plot()
plt.show()
data["10"] = data["Close"].rolling(10).mean() #moving average of short periode is most associated with recent change of stock price called fast signal
data["50"] = data["Close"].rolling(50).mean()#moving average with long period reflect the prices changes over long term history called slow signal
data = data.dropna()
print(data.head())
data["Close"].plot(legend=True)
data["10"].plot(legend=True)
data["50"].plot(legend=True)
plt.show()
#strategie of buying when 10>50 we will buy 
data["Shares"] = [1 if data.loc[ei, "10"]>data.loc[ei, "50"] else 0 for ei in data.index] #create new variable 
print(df.head())
#daily profite
data["Close1"] = data["Close"].shift(-1) #close 1 is the close price of tomorrow
data["profit"] = [data.loc[ei , "Close1"] - data.loc[ei, "Close"] if data.loc[ei, "Shares"]== 1 else 0 for ei in data.index ] #daily profit
#if share=1 the daily profit = close1- close if negative we lose money if positive we win 
# if shares =0 we dont have stock at hand 
print(data.head())
data["profit"].plot()
plt.axhline(y=0, color='red')
plt.show()
data["wealth"] = data["profit"].cumsum() #cumulative wealth
print(data.tail())
data["wealth"].plot()
plt.title("Total money you win is {}".format(data.loc[data.index[-2], "wealth"]))
plt.show()
print("total money you win is ", data.loc[data.index[-2], "wealth"])
print("total money you spend is ", data.loc[data.index[0], "Close"])
#daily return 
data["LogReturn"] = np.log(data["Close"]).shift(-1) - np.log(data["Close"])
data["LogReturn"].hist(bins = 50)
plt.show()
