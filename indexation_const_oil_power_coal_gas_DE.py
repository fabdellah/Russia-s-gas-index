# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:12:50 2018

@author: fabdellah
"""


import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from scipy.optimize import differential_evolution
from dateutil.relativedelta import relativedelta
from scipy import optimize
from numpy import linalg as LA
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore")
from time import time
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


# Import data
file = 'External_Data.xls'
df = pd.read_excel(file)
df_subset = df.dropna(how='any')
row = df_subset.shape[1]
col = df_subset.shape[0]
df_subset = df_subset[3:col]
russia_Index = df_subset[['Data Type','USD.20']].reset_index()[2:df_subset.shape[0]]
russia_Index.drop('index', axis=1, inplace=True)
russia_Index = russia_Index.reset_index()
russia_Index.drop('index', axis=1, inplace=True)
russia_Index.columns = ['date', 'index']

start_date_index = '2014M1'                                                                 # Select dates for Russia's index
end_date_index = '2016M1'
start_indices = list(np.where(russia_Index["date"] == start_date_index)[0])[0]
end_indices = list(np.where(russia_Index["date"] == end_date_index)[0])[0]
yy = russia_Index.iloc[start_indices:end_indices ]
y = yy['index']



start_date_str = '2013-12-31 00:00:00'                                                       # Select dates for commodities prices
end_date_str = '2015-12-31 00:00:00'

start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
df = pd.read_excel('spot_prices.xls')
df_oil = df[['date_oil', 'oil']]
df_oil.columns = ['date', 'oil']
df_oil['date'] = pd.to_datetime(df_oil['date'])  
mask = (df_oil['date'] >= start_date) & (df_oil['date'] <= end_date)
df_oil_x = df_oil.loc[mask].reset_index()  
df_oil_x.drop('index', axis=1, inplace=True) 
df_oil_x = df_oil_x.set_index('date')
df_oil_monthly = df_oil_x.resample("M", how='mean').reset_index().iloc[1:13,:]    

df_power = df[['date_power', 'power']]
df_power.columns = ['date', 'power']
df_power['date'] = pd.to_datetime(df_power['date'])  
mask = (df_power['date'] >= start_date) & (df_power['date'] <= end_date)
df_power_x = df_power.loc[mask].reset_index()  
df_power_x.drop('index', axis=1, inplace=True) 
df_power_x = df_power_x.set_index('date')
df_power_monthly = df_power_x.resample("M", how='mean').reset_index().iloc[1:13,:]    

df_coal = df[['date_coal', 'coal']]
df_coal.columns = ['date', 'coal']
df_coal['date'] = pd.to_datetime(df_coal['date'])  
mask = (df_coal['date'] >= start_date) & (df_coal['date'] <= end_date)
df_coal_x = df_coal.loc[mask].reset_index()  
df_coal_x.drop('index', axis=1, inplace=True) 
df_coal_x = df_coal_x.set_index('date')
df_coal_monthly = df_coal_x.resample("M", how='mean').reset_index().iloc[1:13,:]    

df_gas = df[['date_gas', 'gas']]
df_gas.columns = ['date', 'gas']
df_gas['date'] = pd.to_datetime(df_gas['date'])  
mask = (df_gas['date'] >= start_date) & (df_gas['date'] <= end_date)
df_gas_x = df_gas.loc[mask].reset_index()  
df_gas_x.drop('index', axis=1, inplace=True) 
df_gas_x = df_gas_x.set_index('date')
df_gas_monthly = df_gas_x.resample("M", how='mean').reset_index().iloc[1:13,:]   

df_monthly = np.c_[df_oil_monthly['oil'] , df_power_monthly['power'] , df_coal_monthly['coal'] , df_gas_monthly['gas'] ]



############################## df_monthly_test


start_date_str = '2013-12-31 00:00:00'                                                       # Select dates for commodities prices
end_date_str = '2015-12-31 00:00:00'

start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
df_test = pd.read_excel('spot_prices.xls')
df_oil_test = df_test[['date_oil', 'oil']]
df_oil_test.columns = ['date', 'oil']
df_oil_test['date'] = pd.to_datetime(df_oil_test['date'])  
mask = (df_oil_test['date'] >= start_date) & (df_oil_test['date'] <= end_date)
df_oil_x_test = df_oil_test.loc[mask].reset_index()  
df_oil_x_test.drop('index', axis=1, inplace=True) 
df_oil_x_test = df_oil_x_test.set_index('date')
df_oil_monthly_test = df_oil_x_test.resample("M", how='mean').reset_index().iloc[1:13,:]    

df_power_test = df_test[['date_power', 'power']]
df_power_test.columns = ['date', 'power']
df_power_test['date'] = pd.to_datetime(df_power_test['date'])  
mask = (df_power_test['date'] >= start_date) & (df_power_test['date'] <= end_date)
df_power_x_test = df_power_test.loc[mask].reset_index()  
df_power_x_test.drop('index', axis=1, inplace=True) 
df_power_x_test = df_power_x_test.set_index('date')
df_power_monthly_test = df_power_x_test.resample("M", how='mean').reset_index().iloc[1:13,:]    

df_coal_test = df_test[['date_coal', 'coal']]
df_coal_test.columns = ['date', 'coal']
df_coal_test['date'] = pd.to_datetime(df_coal_test['date'])  
mask = (df_coal_test['date'] >= start_date) & (df_coal_test['date'] <= end_date)
df_coal_x_test = df_coal_test.loc[mask].reset_index()  
df_coal_x_test.drop('index', axis=1, inplace=True) 
df_coal_x_test = df_coal_x_test.set_index('date')
df_coal_monthly_test = df_coal_x_test.resample("M", how='mean').reset_index().iloc[1:13,:]    

df_gas_test = df_test[['date_gas', 'gas']]
df_gas_test.columns = ['date', 'gas']
df_gas_test['date'] = pd.to_datetime(df_gas_test['date'])  
mask = (df_gas_test['date'] >= start_date) & (df_gas_test['date'] <= end_date)
df_gas_x_test = df_gas_test.loc[mask].reset_index()  
df_gas_x_test.drop('index', axis=1, inplace=True) 
df_gas_x_test = df_gas_x_test.set_index('date')
df_gas_monthly_test = df_gas_x_test.resample("M", how='mean').reset_index().iloc[1:13,:]   

df_monthly_test = np.c_[df_oil_monthly_test['oil'] , df_power_monthly_test['power'] , df_coal_monthly_test['coal'] , df_gas_monthly_test['gas'] ]

X_monthly_test = np.c_[np.ones(12), preprocessing.scale(df_monthly_test)]

####################################



def standardize(x):
    """Standardize the original data set."""
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_mse_func(y, x1,x2,x3,x4, coef1,coef2,coef3,coef4):
    """compute the loss by mse."""
    e = y - (x1*coef1 + x2*coef2 + x3*coef3 + x4*coef4)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_mse(y, x, coef):
    """compute the loss by mse."""
    e = y - np.dot(x,coef)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - np.dot(tx,w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def predict(tx,coef):
    return np.dot(tx,coef)
    
   
def polynomial(X):    
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    return poly.fit_transform(X)  


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear regression."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
    return loss, w

def score(X_train,y_train, X_test, y_test,coef):
    y_pred_train = np.dot(X_train,coef)     
    r2 = r2_score(y_test, np.dot(X_test,coef))  
    r2_train = r2_score(y_train, y_pred_train)
    return r2,r2_train


def ridge_regression(X_train,y_train, X_test, y_test):    
    """Ridge regression algorithm."""
    # select the best alpha with RidgeCV (cross-validation)
    # alpha=0 is equivalent to linear regression
    alpha_range = 10.**np.arange(-2, 3)
    ridgeregcv = RidgeCV(alphas=alpha_range, normalize=False, scoring='mean_squared_error') 
    ridgeregcv.fit(X_train, y_train)
    #print('best alpha=',ridgeregcv.alpha_)
    #print('ridgeregcv.coef: ',ridgeregcv.coef_)
    # predict method uses the best alpha value
    y_pred = ridgeregcv.predict(X_test)
    err = metrics.mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  
    r2_train = r2_score(y_train, ridgeregcv.predict(X_train))
    score = ridgeregcv.score
    return ridgeregcv.coef_ , err, r2, r2_train, score
 

def linear_regression(X_train,y_train, X_test, y_test):
    
    regr = linear_model.LinearRegression() 
    regr.fit(X_train, y_train)  
    y_pred = regr.predict(X_test)    
    err =  mean_squared_error(y_test,y_pred)  
    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, regr.predict(X_train))
    return regr.coef_ , err, r2, r2_train
    


def OLS_stat(X,y):
    """Summary statistics for OLS."""
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())





def MA_func_vect_out(nbr_months_per_data, df1, start_date_str,end_date_str, lag, ma_period,reset_period,vect): #vect=np.empty(0) 
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
    start_date_x = start_date - relativedelta(months=int(round(lag))) - relativedelta(months=int(round(ma_period) ))            
    end_date =  end_date + relativedelta(months=int(round(lag))) + relativedelta(months=int(round(ma_period) ))+ relativedelta(months=1) 
    df1['date'] = pd.to_datetime(df1['date'])  
    mask = (df1['date'] >= start_date_x) & (df1['date'] <= end_date)
    df_x = df1.loc[mask].reset_index()  
    df_x.drop('index', axis=1, inplace=True)    
    df_x.iloc[:, [1]]= df_x.iloc[:, [1]].astype(float)   
    ma_monthly = pd.rolling_mean(df_x.set_index('date').resample('1BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[0:nbr_months_per_data, [1]].values
    ma_vect = [ ma_monthly[i] for i in range(0,nbr_months_per_data,int(round(reset_period))) ]
    nbr_reset_periods = int(math.ceil(nbr_months_per_data/int(round(reset_period))))  
    vect = np.empty(0)
    for i in range(0,nbr_reset_periods):
        vect = np.append(vect , ma_vect[i]*np.ones(int(round(reset_period))))    
    vect = vect[0:nbr_months_per_data]    
    return vect
     

def MA_plot(df1, start_date_str,end_date_str, lag, ma_period,reset_period):    
    nbr_months_per_data = 24
    vect = MA_func_vect_out(df1, start_date_str,end_date_str, lag, ma_period,reset_period,np.empty(0))
    time = np.arange(0,nbr_months_per_data)
    plt.plot(time, vect)   
    plt.title('Moving average: Lag = '+str(lag)+ ' MA period = '+ str(ma_period)+ ' Reset period = '+ str(reset_period)  )
    plt.xlabel('Time')
    plt.ylabel('Average level')
    plt.show()







class class_alternate(object):
    
    def __init__(self, df_oil, df_power, df_coal, df_gas, y, start_date_str,end_date_str, nbr_months_per_data, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef):
            self.max_lag = max_lag
            self.max_ma_period = max_ma_period
            self.max_reset_period = max_reset_period
            
            self.nbr_iterations = nbr_iterations
 
            self.nbr_months_per_data = nbr_months_per_data
            self.bounds = [(0, self.max_lag),(0, self.max_lag),(0, self.max_lag),(0, self.max_lag), (1, self.max_ma_period), (1, self.max_ma_period),(1, self.max_ma_period),(1, self.max_ma_period),(1, self.max_reset_period),(1, self.max_reset_period),(1, self.max_reset_period),(1, self.max_reset_period)]
              
            self.df_oil = df_oil
            self.df_power = df_power
            self.df_coal = df_coal
            self.df_gas = df_gas
            self.y = y
            
            self.start_date_str = start_date_str                                             #start_date_str = '2016-01-31 00:00:00'
            self.start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") 
            self.end_date_str = end_date_str                                                 #end_date_str   = '2017-01-31 00:00:00'
            self.end_date = datetime.strptime(self.end_date_str,"%Y-%m-%d %H:%M:%S") 
            
            self.init_coef = init_coef 
            
            
    def MA_func_vect(self, lag_oil, lag_power, lag_coal, lag_gas, ma_period_oil, ma_period_power, ma_period_coal, ma_period_gas, reset_period_oil, reset_period_power, reset_period_coal, reset_period_gas ,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
        """Returns the input matrix computed with the optimal lag, ma_period and reset_period. This matrix is used for the regression in process 2."""
        start_date_oil = self.start_date - relativedelta(months=int(round(lag_oil))) - relativedelta(months=int(round(ma_period_oil) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_oil))) + relativedelta(months=int(round(ma_period_oil) ))+ relativedelta(months=1) 
        self.df_oil['date'] = pd.to_datetime(self.df_oil['date'])  
        mask = (self.df_oil['date'] >= start_date_oil) & (self.df_oil['date'] <= end_date)
        df_xoil = self.df_oil.loc[mask].reset_index()  
        df_xoil.drop('index', axis=1, inplace=True)    
        df_xoil.iloc[:, [1]] = df_xoil.iloc[:, [1]].astype(float)          
                
        start_date_power = self.start_date - relativedelta(months=int(round(lag_power))) - relativedelta(months=int(round(ma_period_power) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_power))) + relativedelta(months=int(round(ma_period_power) ))+ relativedelta(months=1) 
        self.df_power['date'] = pd.to_datetime(self.df_power['date'])  
        mask = (self.df_power['date'] >= start_date_power) & (self.df_power['date'] <= end_date)
        df_xpower = self.df_power.loc[mask].reset_index()  
        df_xpower.drop('index', axis=1, inplace=True)    
        df_xpower.iloc[:, [1]] = df_xpower.iloc[:, [1]].astype(float)            
    
        start_date_coal = self.start_date - relativedelta(months=int(round(lag_coal))) - relativedelta(months=int(round(ma_period_coal) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_coal))) + relativedelta(months=int(round(ma_period_coal) ))+ relativedelta(months=1) 
        self.df_coal['date'] = pd.to_datetime(self.df_coal['date'])  
        mask = (self.df_coal['date'] >= start_date_coal) & (self.df_coal['date'] <= end_date)
        df_xcoal = self.df_coal.loc[mask].reset_index()  
        df_xcoal.drop('index', axis=1, inplace=True)    
        df_xcoal.iloc[:, [1]] = df_xcoal.iloc[:, [1]].astype(float)          
      
        start_date_gas = self.start_date - relativedelta(months=int(round(lag_gas))) - relativedelta(months=int(round(ma_period_gas) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_gas))) + relativedelta(months=int(round(ma_period_gas) ))+ relativedelta(months=1) 
        self.df_gas['date'] = pd.to_datetime(self.df_gas['date'])  
        mask = (self.df_gas['date'] >= start_date_gas) & (self.df_gas['date'] <= end_date)
        df_xgas = self.df_gas.loc[mask].reset_index()  
        df_xgas.drop('index', axis=1, inplace=True)    
        df_xgas.iloc[:, [1]] = df_xgas.iloc[:, [1]].astype(float)             
        
        ma_monthly_oil = pd.rolling_mean(df_xoil.set_index('date').resample('1BM'),window=int(round(ma_period_oil))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_power = pd.rolling_mean(df_xpower.set_index('date').resample('1BM'),window=int(round(ma_period_power))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_coal = pd.rolling_mean(df_xcoal.set_index('date').resample('1BM'),window=int(round(ma_period_coal))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_gas = pd.rolling_mean(df_xgas.set_index('date').resample('1BM'),window=int(round(ma_period_gas))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
                
        ma_vect_oil = [ ma_monthly_oil[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_oil ))) ]
        ma_vect_power = [ ma_monthly_power[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_power ))) ]
        ma_vect_coal = [ ma_monthly_coal[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_coal ))) ]
        ma_vect_gas = [ ma_monthly_gas[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_gas ))) ]
       
        nbr_reset_periods_oil = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_oil))))  
        nbr_reset_periods_power = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_power))))
        nbr_reset_periods_coal = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_coal))))
        nbr_reset_periods_gas = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_gas))))
               
        vect_oil = np.empty(0)
        for i in range(0,nbr_reset_periods_oil):
            vect_oil = np.append(vect_oil , ma_vect_oil[i]*np.ones(int(round(reset_period_oil))))      
        
        vect_power = np.empty(0)
        for i in range(0,nbr_reset_periods_power):
            vect_power = np.append(vect_power , ma_vect_power[i]*np.ones(int(round(reset_period_power))))      
       
        
        vect_coal = np.empty(0)
        for i in range(0,nbr_reset_periods_coal):
            vect_coal = np.append(vect_coal , ma_vect_coal[i]*np.ones(int(round(reset_period_coal))))      

        
        vect_gas = np.empty(0)
        for i in range(0,nbr_reset_periods_gas):
            vect_gas = np.append(vect_gas , ma_vect_gas[i]*np.ones(int(round(reset_period_gas))))      

        vect_oil = vect_oil[0:self.nbr_months_per_data]
        vect_power = vect_power[0:self.nbr_months_per_data]
        vect_coal = vect_coal[0:self.nbr_months_per_data]
        vect_gas = vect_gas[0:self.nbr_months_per_data]
        
        vect = np.c_[np.array(vect_oil) ,np.array(vect_power) , np.array(vect_coal) , np.array(vect_gas) ]
        
        return vect
    
        
    def func_lag_period(self, parameters, *data):
        """Objective function to minimize."""
        lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = parameters
        df_oil , df_power , df_coal , df_gas , coef, values = data
        values = compute_mse(preprocessing.scale(self.y) ,np.c_[np.ones(self.nbr_months_per_data), preprocessing.scale(self.MA_func_vect(lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas, np.empty(0)))]   , coef) 
        return values


    def de_optimization(self, coef):
        """Differential evolution for the lag, ma_period and reset_period for oil, power, coal and gas."""        
        args = (self.df_oil,self.df_power,self.df_coal,self.df_gas, coef, np.empty(1))
        result =differential_evolution(self.func_lag_period,  self.bounds, args=args)
        lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = result.x                          
         
        return lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas
    
    
    def alternate(self):
        """Alternate between process 1 and process 2."""    
        max_iters = 100
        gamma = 0.1
        coef =  self.init_coef
        gradient_w = self.init_coef      
        
        for itera in range(self.nbr_iterations):            
            print('///////////////////////////////////////')
            print('Iteration: ', itera )
            
            #process 1: optimal lag, optimal ma_period and optimal reset period for a given coefficient           
            t00 = time()
            lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = self.de_optimization(coef)
            t11 = time()
            d1 = t11-t00
            lag = np.array([lag_oil, lag_power, lag_coal, lag_gas])
            ma_period = np.array([period_oil, period_power, period_coal, period_gas])
            reset_period = np.array([reset_oil, reset_power, reset_coal, reset_gas])
            print ("Duration of process 1 in Seconds %6.3f" %d1)      
            
            #process2: optimal coefficient for a given lag, ma_period and reset period
            t02 = time()
            X_df = self.MA_func_vect(lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas, np.empty(0))     
            XX_stand = np.c_[np.ones(X_df.shape[0]), preprocessing.scale(X_df).reshape(self.nbr_months_per_data,4)] 
            w_initial = gradient_w
            X_train, X_test, y_train, y_test = train_test_split(XX_stand, preprocessing.scale(self.y), random_state=1)
            
            gradient_loss, gradient_w = gradient_descent(preprocessing.scale(y_train), X_train, w_initial, max_iters, gamma)  
            res_ridge = ridge_regression(X_train, y_train, X_test, y_test)
             
            # update coef
            coef = res_ridge[0]
            y_pred_GD = np.dot(X_test,gradient_w)
            print('--------- Gradient descent ---------')
            print('Coef GD:', gradient_w )
            print('Error GD:', metrics.mean_squared_error(y_test, y_pred_GD))
            print('R2_train GD', r2_score(preprocessing.scale(y_train), np.dot(X_train,gradient_w))  )
            print('R2_test GD', r2_score(y_test, y_pred_GD)  )
                                
            print('--------- Ridge regression ---------')
            
            print('Coef RR:', res_ridge[0] )
            print('Error RR: ', res_ridge[1])
            print('R2_train RR: ', res_ridge[3])
            print('R2_test RR: ', res_ridge[2])
            
            t12 = time()
            d2 = t12-t02
            print ("Duration of process 2 in Seconds %6.3f" % d2)        
                
        return coef , lag , ma_period, reset_period ,X_df, XX_stand, X_train, X_test, y_train, y_test
           


if __name__ == '__main__':
    
    #Step 1: optimizing lag, ma_period, reset_period and get the coefficients     
    
                                   #df_oil, df_power, df_coal, df_gas, y, start_date_str,end_date_str, nbr_months_per_year, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef
    optimization = class_alternate(df_oil, df_power ,df_coal ,df_gas,y ,'2014-01-31 00:00:00','2016-01-31 00:00:00', 24, 36, 12 , 12, 7, np.array([0,0,0,0,0]))
    t0 = time()
    coef , lag , ma_period, reset_period , X_df, XX_stand, X_train, X_test, y_train, y_test = optimization.alternate()    
    t1 = time()
    d = t1 - t0
    print ("Total duration in Seconds %6.3f" % d)               
    print('final coef: ', coef)
    
    
    
    OLS_stat(XX_stand,preprocessing.scale(y))
    
    
    
    
    
    
    # Testing 
    
    start_date_index = '2016M1'                                                                
    end_date_index = '2017M1'
    nbr_months_per_testing_data = 12   
    start_indices = list(np.where(russia_Index["date"] == start_date_index)[0])[0]
    end_indices = list(np.where(russia_Index["date"] == end_date_index)[0])[0]
    yy_test = russia_Index.iloc[start_indices:end_indices ]
    y_test = yy_test['index']
    
    
    start_day = '2016-01-31 00:00:00'
    end_day = '2017-01-31 00:00:00'
    oil_test = MA_func_vect_out(df_oil, start_day , end_day , lag[0] , ma_period[0] , reset_period[0] , np.empty(0))
    power_test = MA_func_vect_out(df_power, start_day , end_day , lag[1],ma_period[1], reset_period[1] , np.empty(0))
    coal_test = MA_func_vect_out(df_coal, start_day , end_day , lag[2], ma_period[2], reset_period[2] , np.empty(0))
    gas_test = MA_func_vect_out(df_gas, start_day , end_day , lag[3],ma_period[3], reset_period[3] , np.empty(0))
    
    X_test = np.c_[oil_test, power_test, coal_test, gas_test]
    X_test_stand = np.c_[np.ones(X_test.shape[0]), preprocessing.scale(X_test).reshape(nbr_months_per_testing_data,4)]   
    OLS_stat(X_test_stand ,preprocessing.scale(y_test))
    
    error_test = compute_mse(preprocessing.scale(y), X_test_stand, coef)
    
    
    score(XX_stand,preprocessing.scale(y), X_test_stand, preprocessing.scale(y_test),coef )
    
    
    
    
    
    
    #Step 2: Model calibration
    
    #Step 3: Monte Carlo for gas storage
   
