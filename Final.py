# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:43:20 2021

@author: Rawabi
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import dateutil.parser
from pprint import pprint
import json
import time
import sys
import re
import datetime

# APIs
import quandl



# Quandl API Calls
df_price = pd.read_csv('https://www.quandl.com/api/v3/datasets/BNC3/GWA_BTC.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # Price, volume
df_eth = pd.read_csv('https://www.quandl.com/api/v3/datasets/GDAX/ETH_USD.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # ETH Price, volume
df_fees = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/TRFUS.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # Txn fees
df_cost = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/CPTRA.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # cost per txn
df_no = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/NTRAN.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # num txns
df_noblk = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/NTRBL.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # txns per block
df_blksz = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/AVBLS.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # blk size
df_unq = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/NADDU.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # unique addys
df_hash = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/HRATE.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # hash rate
df_diff = pd.read_csv('https://www.quandl.com/api/v3/datasets/BCHAIN/DIFF.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # difficulty

df_nasdaq = pd.read_csv('https://www.quandl.com/api/v3/datasets/NASDAQOMX/COMP.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # NASDAQ Composite
df_nasdaq = df_nasdaq.rename(columns={'Trade Date': 'Date','Index Value':'Nasdaq'})
df_nasdaq = df_nasdaq.drop(['High','Low','Total Market Value','Dividend Market Value'], 1)

df_gold = pd.read_csv('https://www.quandl.com/api/v3/datasets/NASDAQOMX/QGLD.csv?api_key=6pB7KEsRUxpRHzMKgEsh') # Nasdaq GOLD Index
df_gold = df_gold.rename(columns={'Trade Date': 'Date','Index Value':'Gold'})
df_gold = df_gold.drop(['High','Low','Total Market Value','Dividend Market Value'], 1)


# Helper functions
def to_date(datestring):
    date = dateutil.parser.parse(datestring)
    return date

def list_to_average(list):
    try:
        avg = list[0]/list[1]
    except:
        avg = 0
    return avg

def to_log(num):
    return np.log(num)



df = df_price.drop('Open', 1)
df = df.drop(['High','Low'], 1)

df = df.rename(columns={'Close': 'BTCPrice','Volume':'BTCVol'})
df = df_eth.merge(df,how='inner',on='Date')
df = df.rename(columns={'Open': 'ETHPrice'})
df = df.drop(['High','Low'], 1)
df = df_fees.merge(df, how='inner', on='Date')
df = df.rename(columns={'Value': 'TxFees'})
df = df_cost.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'CostperTxn'})
df = df_no.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'NoTxns'})
df = df_noblk.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'NoperBlock'})
df = df_blksz.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'AvgBlkSz'})
df = df_unq.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'UniqueAddresses'})
df = df_hash.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'HashRate'})
df = df_diff.merge(df,how='inner',on='Date')
df = df.rename(columns={'Value': 'Difficulty'})

df = df_nasdaq.merge(df,how='inner',on='Date')
df = df_gold.merge(df,how='inner',on='Date')


ct = [i for i in reversed(range(len(df)))]
df['DateNum'] = ct 

df['Date'] = df['Date'].apply(to_date)
df['Date'] = pd.to_datetime(df['Date'])
df['Date2'] = df['Date']
df = df.set_index('Date2')

df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Weekday'] = df['Date'].dt.weekday




df = df[['BTCPrice','ETHPrice','BTCVol','TxFees','CostperTxn','NoTxns','NoperBlock','AvgBlkSz','UniqueAddresses',
         'HashRate','Difficulty','Nasdaq','Gold','DateNum','Date','Month','Quarter','Weekday']]
df_hist = df

print(df_hist.shape)
df_hist.info()

df_hist.head()
df_hist.tail()


# run coinmarketcap_hist.py weekly to generate .json file 
mkt_cap = pd.read_json('D:\\Bitcoin_Project\\coinmarketcap_hist2.json').T
mkt_cap['Date'] = mkt_cap.index
mkt_cap['Date'] = pd.to_datetime(mkt_cap['Date'],format='%Y%m%d',errors='ignore')
mkt_cap.head()
mkt_cap = mkt_cap.set_index('Date')
mkt_cap = mkt_cap[['BTC','ETH']]
mkt_cap.tail()



df_goog = pd.read_csv('D:\\Bitcoin_Project\\multiTimeline (2).csv') # Google Trends "bitcoin" interest over time 
df_goog = df_goog.iloc[2:]
df_goog = df_goog.rename(columns={'Category: All categories': 'Interest'})
df_goog['Date2'] = df_goog.index
df_goog['Date2'] = pd.to_datetime(df_goog['Date2'])
df_goog = df_goog.set_index('Date2')
df_goog.info()
df_goog.head()

df_mc = pd.concat([mkt_cap, df_goog], axis=1)
df_mc.head()

df_all = pd.concat([df_hist, df_mc], axis=1)

df_all = df_all.fillna(method='ffill')
df_all = df_all.iloc[200:,:]
df_all.head()

df_all = df_all[['BTCPrice','ETHPrice','BTCVol', 'CostperTxn','TxFees','NoTxns','AvgBlkSz','UniqueAddresses','HashRate','Difficulty','Nasdaq','Gold','Interest','DateNum','Quarter','Month','Weekday']]
df_all = pd.DataFrame(df_all,dtype=np.float64) # convert all values to float64


# add log columns
df_all['logBTCPrice'] = df_all['BTCPrice'].apply(to_log)
df_all['logNasdaq'] = df_all['Nasdaq'].apply(to_log)
df_all['logETHPrice'] = df_all['ETHPrice'].apply(to_log)
df_all['logGold'] = df_all['Gold'].apply(to_log)

#df_all['logCrypto Market Cap'] = df_all['Crypto Market Cap'].apply(to_log)
df_all['logInterest'] = df_all['Interest'].apply(to_log)
df_all['logCostperTxn'] = df_all['CostperTxn'].apply(to_log)
df_all['logTxFees'] = df_all['TxFees'].apply(to_log)
df_all['logNoTxns'] = df_all['NoTxns'].apply(to_log)
df_all['logAvgBlkSz'] = df_all['AvgBlkSz'].apply(to_log)
df_all['logUniqueAddresses'] = df_all['UniqueAddresses'].apply(to_log)
df_all['logHashRate'] = df_all['HashRate'].apply(to_log)
df_all['logBTCVol'] = df_all['BTCVol'].apply(to_log)
df_all['logDifficulty'] = df_all['Difficulty'].apply(to_log)

df_all.columns

# pickle the consolidate DataFrame
df_all.to_pickle('D:\\Bitcoin_Project//benson_btcsentiment_df.pkl')

#EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time
import sys
import re
import datetime


# Helper functions
def str_to_int(str):
    str2 = int(str.replace('$','').replace(',',''))
    return str2

def to_currency(int):
    return '{:,}'.format(int)

# def thousands(int):
#     divided = int // 1000
#     return to_currency(divided)

# def y_fmt(x, y):
#     return '{:2.2e}'.format(x).replace('e', 'x10^')


df = pd.read_pickle('D:\\Bitcoin_Project//benson_btcsentiment_df.pkl')
df = df[['BTCPrice','ETHPrice','BTCVol','TxFees','CostperTxn','NoTxns','AvgBlkSz','UniqueAddresses','HashRate','Nasdaq','Gold','Interest']]
df_all = df
df_hist = df
df.head()

df_all.corr().sort_values('BTCPrice')

df_all = df
y1 = pd.Series(df_all['BTCPrice'])
y2 = pd.Series(df_all['Interest'])
x = pd.Series(df_all.index.values)

fig, ax = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()

ax.plot(x,y1,'b')
ax2.plot(x,y2,'g')
ax.set_ylabel("Price $US",color='b',fontsize=12)
ax2.set_ylabel("Google Search Interest",color='g',fontsize=12)
ax.grid(True)
plt.title("Bitcoin Price vs. Google Search Interest", fontsize=14)
ax.set_xlabel('Date', fontsize=12)
fig.autofmt_xdate()

plt.savefig('D:\\Bitcoin_Project\\charts\googlesearchinterest.png')
print(plt.show())


# BTC Price vs Nasdaq
y2 = pd.Series(df_all['Nasdaq'])

fig, ax = plt.subplots()

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=12,color='blue')
ax2.set_ylabel("Nasdaq Composite Index",fontsize=12,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Nasdaq Composite Index", fontsize=14,color='black')
ax.set_xlabel('Date', fontsize=12, color='black')
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/nasdaq.png')
print(plt.show())


# BTC Price vs ETH Price
y2 = pd.Series(df_all['ETHPrice'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Ethereum Price",fontsize=14,color='green')
# ax.grid(True)
plt.title("Bitcoin Price vs. Ethereum Price", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/ethprice.png')
print(plt.show())

# BTC Price vs ETH Price
y2 = pd.Series(df_all['CostperTxn'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Cost Per Transaction",fontsize=14,color='green')
# ax.grid(True)
plt.title("Bitcoin Price vs. Cost Per Transaction", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
# plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/costpertxn.png')
print(plt.show())


# BTC Price vs Volume
df_all = df_all[:365]

y1 = pd.Series(df_all['BTCPrice'])
y2 = pd.Series(df_all['BTCVol'])
x = pd.Series(df_all.index.values)

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Volume",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Volume", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/fig1.png')
print(plt.show())

# BTC Price vs Transaction Fees
y2 = pd.Series(df_all['TxFees'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("TxFees",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Transaction Fees", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/fig2.png')
print(plt.show())

# BTC Price vs Cost per Transaction
y2 = pd.Series(df_all['CostperTxn'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("CostperTxn",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Cost per Transaction", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/fig3.png')
print(plt.show())

# BTC Price vs Number of Transactions
y2 = pd.Series(df_all['NoTxns'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("NumberofTxns",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Number of Transactions", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/fig4.png')
print(plt.show())


# BTC Price vs Block Size
y2 = pd.Series(df_all['AvgBlkSz'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Average Block Size",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Average Block Size", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/fig6.png')
print(plt.show())

# BTC Price vs Unique Addresses
y2 = pd.Series(df_all['UniqueAddresses'])

ax = plt.gca()
ax2 = ax.twinx()
#plt.axis('normal')
ax.plot(x,y1, 'b',linewidth=1.5)
ax2.plot(x,y2, 'g',linewidth=1.5)
ax.set_ylabel("Price",fontsize=14,color='blue')
ax2.set_ylabel("Unique Addresses",fontsize=14,color='green')
ax.grid(True)
plt.title("Bitcoin Price vs. Unique Addresses", fontsize=20,color='black')
ax.set_xlabel('Date', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/fig7.png')
print(plt.show())



df = df_all
y = pd.Series(df['BTCPrice'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCPriceQuandl.png')
print(plt.show())

df = df[:365]
y = pd.Series(df['BTCPrice'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, LTM",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/LTMBTCPriceQuandl.png')
print(plt.show())

df = df[:90]
y = pd.Series(df['BTCPrice'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Closing Price, Last 90 Days",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/90DBTCPriceQuandl.png')
print(plt.show())


df = df_all
y = pd.Series(df['TxFees'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Fees (USD), Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Fees', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCTxnFeesQuandl.png')
plt.show()

df = df[:365]
y = pd.Series(df['TxFees'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Fees (USD), LTM",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Fees', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/LTMBTCTxnFeesQuandl.png')
plt.show()

df = df[:90]
y = pd.Series(df['TxFees'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Fees (USD), Last 90 Days",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Transaction Fees', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/90DBTCTxnFeesQuandl.png')
plt.show()

df = df_all
y = pd.Series(df['CostperTxn'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Cost Per Transaction (BTC?), Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cost per Transaction', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCCostperTxnQuandl.png')
plt.show()


df = df_all
y = pd.Series(df['BTCVol'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Transaction Volume (USD), Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCTxnVolQuandl.png')
plt.show()

df = df_hist
y = pd.Series(df['NoTxns'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Number of Bitcoin Transactions, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCTxnAmtQuandl.png')
plt.show()


df = df_all
y = pd.Series(df['AvgBlkSz'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Average Block Size, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Block Size', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCAvgBlockSizeQuandl.png')
plt.show()

# Quandl: Unique BTC Addresses
df = df_all
y = pd.Series(df['UniqueAddresses'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Number of Unique Bitcoin Addresses, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Unique Addresses', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCNoAddressesQuandl.png')
plt.show()


# Quandl: BTC Hash Rate
df = df_all
y = pd.Series(df['HashRate'])
x = pd.Series(df.index.values)

plt.plot(x,y)
plt.title("Bitcoin Hash Rate, Historical to Date",fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Hash Rate', fontsize=12)
plt.tight_layout()
plt.savefig('D:\\Bitcoin_Project\\charts/HistBTCHashRateQuandl.png')
plt.show()

#LR Analysis

import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings; warnings.simplefilter('ignore')
import json
import time
import sys
import re
import datetime

df = pd.read_pickle('D:\Bitcoin_Project/benson_btcsentiment_df.pkl')
df = df[['BTCPrice','logBTCPrice','logETHPrice','logBTCVol','logTxFees','logCostperTxn','logNoTxns','logAvgBlkSz','logUniqueAddresses','logHashRate','logNasdaq','logGold','logInterest','Interest','TxFees','Nasdaq']]
df_all = df
df_hist = df
df.head(2)

# evaluate the correlation of the majority of features
df_all.corr().sort_values('logBTCPrice')

sns.set(style="darkgrid", color_codes=True)
sns.pairplot(df_all)
plt.savefig('D:\Bitcoin_Project/pairplotfeatureuniverse.png')

#The Core 3-Feature Model

# Reducing to few key features
df_all = df_all[['logBTCPrice','logNasdaq','logInterest','logTxFees']]
#df_all = df_all[['logBTCPrice','logNasdaq','logTxFees']]
df_all.corr().sort_values('logBTCPrice')

sns.set(style="darkgrid", color_codes=True)
sns.pairplot(df_all,plot_kws={'alpha':0.3})
plt.title('Correlation by Feature',fontsize=14)
plt.savefig('D:\Bitcoin_Project\modelpairplot2.png')

df = df_all
# STATSMODELS
# Feature matrix (X) and target vector (y)
y, X = patsy.dmatrices('logBTCPrice ~ logInterest + logNasdaq + logTxFees', data=df, return_type="dataframe")

model = sm.OLS(y,X)
fit = model.fit()
fit.summary()

# SKLEARN
lr = LinearRegression()


X = df[['logInterest','logNasdaq','logTxFees']]
# Choose the response variable(s)
y = df['logBTCPrice']

lr.fit(X,y)
# Print out the R^2 for the model against the full dataset
print(lr.score(X,y))
print(lr.intercept_)
print(lr.coef_)

# Plotting residuals on a time series basis.
# Note that residuals are not random and will require further adjustments at a later time.  
# fit.resid.plot(style='o');
# plt.ylabel("Residual",fontsize=12)
# plt.xlabel("Time",fontsize=12)
# plt.title('Residual Over Time',fontsize=14)
# plt.savefig('D:\Bitcoin_Project\charts/residovertime.png')


y_pred = lr.predict(X)

residuals = y - y_pred
print(residuals)
sns.distplot(residuals);
plt.ylabel("Frequency",fontsize=12)
plt.xlabel("Residual",fontsize=12)
plt.title('Residual Histogram',fontsize=14)
plt.savefig('D:\Bitcoin_Project\charts/residhist.png')

#from sklearn import cross_validation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

# Fit the model against the training data
lr.fit(X_train, y_train)
# # Evaluate the model against the testing data
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

# sklearn prediction (logBTCPrice); seaborn
# ax = sns.regplot(x=y_test,y=lr.predict(X_test), data=df)
# ax.set(xlabel='Actual (log)', ylabel='Predicted (log)', title = 'Bitcoin Price: Predicted vs. Actual (log)')
# plt.savefig('D:\Bitcoin_Project\charts/logpredictedvsactual.png')



# sklearn prediction (BTCPrice); seaborn
# ax = sns.regplot(x=np.exp(y_test),y=np.exp(lr.predict(X_test)), data=df)
# ax.set(xlabel='Actual, US$', ylabel='Predicted, US$', title = 'Bitcoin Price: Predicted vs. Actual')
# plt.savefig('D:\Bitcoin_Project\charts/predictedvsactual.png')

# limit x and y ticks to 8000 to see model prediction capability vs actual is near-aligned
# ax = sns.regplot(x=np.exp(y_test),y=np.exp(lr.predict(X_test)), data=df)
# ax.set(xlabel='Actual, US$', ylabel='Predicted, US$', title = 'Bitcoin Price: Predicted vs. Actual')
# plt.xlim(0, 8000)
# plt.ylim(0, 8000)
# plt.savefig('D:\Bitcoin_Project\charts/predictedvsactuallimits.png')


# GOOGLE SEARCH INTEREST
# x = df['logBTCPrice']
# y = df['logInterest']

# ax = sns.regplot(x,y, data=df)
# ax.set(xlabel='log Bitcoin Price', ylabel='log Google Search Interest', title = 'Bitcoin Price vs. Google Search Interest')
# plt.savefig('D:\Bitcoin_Project\charts/interestvsbtcprice.png')



# y1 = pd.Series(df_hist['BTCPrice'])
# y2 = pd.Series(df_hist['Interest'])
# x = pd.Series(df_hist.index.values)

# fig, _ = plt.subplots()

# ax = plt.gca()
# ax2 = ax.twinx()

# ax.plot(x,y1,'b')
# ax2.plot(x,y2,'g')
# ax.set_ylabel("Price US$",color='b',fontsize=12)
# ax2.set_ylabel("Google Search Interest",color='g',fontsize=12)
# ax.grid(True)
# plt.title("Bitcoin Price vs. Google Search Interest", fontsize=14)
# ax.set_xlabel('Date', fontsize=12)
# fig.autofmt_xdate()

# plt.savefig('D:\Bitcoin_Project\charts/googlesearchinterest.png')

# print(plt.show())


# y, X = patsy.dmatrices('logBTCPrice ~ logInterest', data=df, return_type="dataframe")

# # Create your model
# model2 = sm.OLS(y,X)
# # Fit your model to your training set
# fit2 = model2.fit()
# # Print summary statistics of the model's performance
# fit2.summary()


# # FEATURE ANALYSIS: NASDAQ COMPOSITE INDEX
# x = df['logBTCPrice']
# y = df['logNasdaq']

# ax = sns.regplot(x,y, data=df)
# ax.set(xlabel='log Bitcoin Price', ylabel='log Nasdaq Composite Index', title = 'Bitcoin Price vs. Nasdaq')
# plt.savefig('D:\Bitcoin_Project\charts/nasdaqvsbtcprice.png')

# y1 = pd.Series(df_hist['BTCPrice'])
# y2 = pd.Series(df_hist['Nasdaq'])
# x = pd.Series(df_hist.index.values)

# fig, _ = plt.subplots()

# ax = plt.gca()
# ax2 = ax.twinx()

# ax.plot(x,y1,'b')
# ax2.plot(x,y2,'g')
# ax.set_ylabel("Price US$",color='b',fontsize=12)
# ax2.set_ylabel("Nasdaq Composite Index",color='g',fontsize=12)
# # ax.grid(True)
# plt.title("Bitcoin Price vs. Nasdaq Composite Index", fontsize=14)
# ax.set_xlabel('Date', fontsize=12)
# fig.autofmt_xdate()
# # ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

# plt.savefig('D:\Bitcoin_Project\charts/nasdaq.png')
# print(plt.show())


# y, X = patsy.dmatrices('logBTCPrice ~ logNasdaq', data=df, return_type="dataframe")

# # Create your model
# model3 = sm.OLS(y,X)
# # Fit your model to your training set
# fit3 = model3.fit()
# # Print summary statistics of the model's performance
# fit3.summary()

# # FEATURE ANALYSIS: TRANSACTION FEES
# x = df['logBTCPrice']
# y = df['logTxFees']

# ax = sns.regplot(x,y, data=df)
# ax.set(xlabel='log Bitcoin Price', ylabel='log Network Transaction Fees', title = 'Bitcoin Price vs. Network Transaction Fees')
# plt.savefig('D:\Bitcoin_Project\charts/txfeesvsbtcprice.png')


# y1 = pd.Series(df_hist['BTCPrice'])
# y2 = pd.Series(df_hist['TxFees'])
# x = pd.Series(df_hist.index.values)

# fig, _ = plt.subplots()

# ax = plt.gca()
# ax2 = ax.twinx()

# ax.plot(x,y1,'b')
# ax2.plot(x,y2,'g')
# ax.set_ylabel("Price US$",color='b',fontsize=12)
# ax2.set_ylabel("Network Transaction Fees",color='g',fontsize=12)
# # ax.grid(True)
# plt.title("Bitcoin Price vs. Network Transaction Fees", fontsize=14)
# ax.set_xlabel('Date', fontsize=12)
# fig.autofmt_xdate()
# # ax.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))

# plt.savefig('D:\Bitcoin_Project\charts/txfees.png')
# print(plt.show())




# y, X = patsy.dmatrices('logBTCPrice ~ logTxFees', data=df, return_type="dataframe")

# # Create your model
# model4 = sm.OLS(y,X)
# # Fit your model to your training set
# fit4 = model4.fit()
# # Print summary statistics of the model's performance
# fit4.summary()

# REGULARIZATION
from sklearn.linear_model import RidgeCV

rcv = RidgeCV(cv=20)

rcv.fit(X_train, y_train)
print(rcv.score(X_train,y_train))
rcv.score(X_test, y_test)