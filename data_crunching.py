import dill
import os
import cloudpickle
import types
import pandas as pd
import numpy as np
import pickle
import warnings
import datetime as dt
from datetime import datetime
from time import strptime
import re
import calendar
import QuantLib as ql
current_dir = os.getcwd()
starting_date = ql.Date(8,11,2019)
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()

moneyness_space = [.9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1] # S_0/K
maturity_space = [30, 45, 60, 90, 120, 180, 240, 300, 360, 420]
S0 = 3093.08
def dt_to_ql(date):
    return ql.Date(date.day, date.month, date.year)
def ql_to_dt(Dateql):
    return dt.datetime(Dateql.year(), Dateql.month(), Dateql.dayOfMonth())
def myround(x, base=5):
    return base * round(x/base)
def closest(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]
chainfile = 'option_chain.csv' # must be in same directory
df = pd.read_csv(chainfile, sep = ";")

date_format = '%d/%m/%Y'
df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
df.dropna(inplace = True)
df = df.iloc[:, :12]
df.drop('Calls', axis = 1, inplace = True)
df['Price'] = df['Ask']
df['ttm'] = (df['Expiration Date'] - ql_to_dt(starting_date)).dt.days
df['Moneyness'] = df.Strike/S0
df.set_index(df['ttm'], inplace = True)
df.drop(['Delta', 'Gamma', 'Vol',
        'Net', 'Last Sale', 'Open Int',
        'Bid', 'Ask', 'ttm'],
         axis = 1, inplace = True)

# df.to_csv(current_dir + '/callchain.csv')

strikes = [myround(S0 * m) for m in moneyness_space]

testdf = df.loc[df.index == 406]

testdf['Deltastrike'] = testdf['Strike'] - testdf['Strike'].shift(1)
testdf = testdf.loc[testdf.Deltastrike == 25]
testdf.drop('Deltastrike', axis = 1, inplace = True)

testdf = testdf.loc[(testdf['Strike'] >= strikes[0]) & (testdf['Strike'] <= strikes[-1])]
strikes = testdf.Strike.values

df = df.loc[df.Strike.isin(strikes)]
df['Deltastrike'] = df['Strike'] - df['Strike'].shift(1)
df = df.loc[(df.Deltastrike == 25) & (df.index >= 16) & (df.index !=119)]

moneys = np.unique(df.Moneyness.values)
moneyspace_list = [m for m  in moneys]
moneyspace = [closest(moneyspace_list, k) for k in moneyness_space]
moneypklname = 'predobj_Moneyness.pkl'


ttm_list = df.index.values.tolist()
ttms = [closest(ttm_list, k) for k in maturity_space]

df = df.loc[(df.index.isin(ttms)) & (df.Moneyness.isin(moneyspace))]

ivs = df['IV']
expiries = df['Expiration Date']

df.drop(['Deltastrike', 'IV','Moneyness', 'Expiration Date'], axis = 1, inplace = True)
df.reset_index(inplace = True)
df.rename(columns = {'Strike':'strikes', 'Price':'price'}, inplace = True)

with open(moneypklname, 'wb') as f:
    pickle.dump(moneyspace, f)
with open('predobj_ttms.pkl', 'wb') as f:
    pickle.dump(ttms, f)

rf = 0.0156 # 3 month current t-bill rate
dr = 0.0188 # october average yield for s&p
v0 = 0.1207**2 # VIX^2

observables = np.array([S0, rf, dr, v0])

s_date = ql_to_dt(starting_date)
obsdict = {s_date:observables}

with open('predobj_observables.pkl', 'wb') as f:
    pickle.dump(obsdict, f)
dfdict = {s_date: df}
dfname = 'predobj_df.pkl'
with open(dfname, 'wb') as f:
    pickle.dump(dfdict, f)

with open('NN_predicted_calibrated_params.pkl', 'rb') as f:
    pred_params = pickle.load(f)
pred_params = list(pred_params.values())[0]
obs = list(obsdict.values())[0]
pd.set_option('use_inf_as_na', True)
ivs = ivs.reset_index()
ivs = ivs.drop('ttm', axis = 1)
ivs = ivs.replace([np.inf, -np.inf], np.nan)
ivs = ivs.dropna()
ivs = np.transpose(ivs['IV'].tolist()).tolist()
ivdict = {s_date:ivs}

with open('predobj_ivs2.pkl', 'wb') as f:
    pickle.dump(ivdict, f)
expiries = expiries.reset_index()
expiries = expiries.set_index('Expiration Date')
# expiries = expiries.drop('ttm', axis = 1)
expiries = expiries.index.strftime('%Y-%m-%d').tolist()
expiries = [dt.datetime.strptime(o, '%Y-%m-%d') for o in expiries]


with open('predobj_expdates.pkl', 'wb') as f:
    pickle.dump(expiries, f)


print('Data prepared')



# with open('predobj_ivs.pkl', 'rb') as f:
#     predivs = pickle.load(f)
# with open('he_ivs.pkl', 'rb') as f:
#     he_ivs = pickle.load(f)
# print(predivs)
