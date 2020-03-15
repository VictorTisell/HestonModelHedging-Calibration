from __future__ import division
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import QuantLib as ql
has_function_evals = True
float_type = np.float32
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
import os
import math
data_dir = os.getcwd()

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from os.path import isfile
from os import getcwd
from copy import deepcopy, copy
import matplotlib.pyplot as plt
from six import string_types
import dill
from functools import partial
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Input
from keras.layers.merge import add
#from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
                            Callback, TensorBoard
from keras import backend as K
from keras.constraints import maxnorm
#from keras import initializations
from keras.initializers import VarianceScaling, RandomUniform
#from keras.utils import Sequence
from keras.utils.vis_utils import plot_model

import warnings
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import types
import data_crunching
dill._dill._reverse_typemap['ClassType'] = type

warnings.simplefilter(action='ignore', category=FutureWarning)
starting_date = ql.Date(8,11,2019)
f_time = dt.datetime.now()
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
calendar_swe = ql.Sweden()
# Option conventions

# moneyness = [.8, .85, .9, .95, .975, .99, 1.0, 1.01, 1.025, 1.05, 1.1, 1.15, 1.2] # S_0/K
# maturities = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420]
with open('predobj_Moneyness.pkl', 'rb') as f:
    moneyness = pickle.load(f)
print(moneyness)
with open('predobj_ttms.pkl', 'rb') as f:
    maturities = pickle.load(f)
print(maturities)
with open('predobj_observables.pkl', 'rb') as f:
    predobj_obs = pickle.load(f)
spot = list(predobj_obs.values())[0][0]
print([spot*m for m in moneyness])
with open('predobj_df.pkl', 'rb') as f:
    predobj_df = pickle.load(f)
he_analytic = {'name' : 'Heston',
               'process' : ql.HestonProcess,
               'model' : ql.HestonModel,
               'engine' : ql.AnalyticHestonEngine,
               'option_type' : ql.Option.Call,
               'transformation' : np.log,
               'inverse_transformation' : np.exp}
# volatility skeleton
def vol_skeleton(strike_space, expiration_space, calc_date, spot_price):
    # ändrat
    strikes = [spot_price *k for k in strike_space]

    expiries = [calc_date + d_time for d_time in expiration_space]
    ttm_d = [(day-calc_date) for day in expiries]
    ttm_y = [day_count.yearFraction(calc_date, day) for day in expiries]

    output_array = np.array((ttm_d, strikes))
    product_vol_surface = list(product(*output_array))
    df = pd.DataFrame(product_vol_surface, columns = ['ttm', 'strikes'])
    return strikes, np.array((ttm_y)), expiries, df


def set_rf(calc_date, rf):
    return ql.YieldTermStructureHandle(ql.FlatForward(calc_date, rf, day_count))
def set_dividend(calc_date, dividend_rate):
    return ql.YieldTermStructureHandle(
            ql.FlatForward(calc_date, dividend_rate, day_count))
def set_spot(spot_price):
    return ql.QuoteHandle(ql.SimpleQuote(spot_price))
def EU_option(calc_date, option_type, strike, ttm):
    cashflow = ql.PlainVanillaPayoff(option_type, strike)
    maturity = calc_date + int(ttm)
    ex = ql.EuropeanExercise(maturity)
    return ql.VanillaOption(cashflow, ex)

class Heston:
    def __init__(self, heston_dict, spot_price = 100., rf = 0.01, dividend_rate = 0.0,
                ivar = 0.1, calc_date = starting_date, expiration_space = maturities,
                strike_space = moneyness, mean_rev = None, eq_var = None, volvol = None, corr = None):
        self.mod_dict = heston_dict
        if ('process' not in self.mod_dict
            or 'model' not in self.mod_dict
            or 'engine' not in self.mod_dict
            or 'name' not in self.mod_dict
            or 'option_type' not in self.mod_dict):
            raise RuntimeError('Missing input parameters in the dictionary')
        self.option_type = self.mod_dict['option_type']
        self.calc_date = calc_date
        ql.Settings.instance().evaluationDate = self.calc_date
        self.no_params = 4
        self.sigma =  volvol
        self.rho = corr
        self.theta = eq_var
        self.kappa = mean_rev

        self.parameters = np.array((self.kappa, self.theta, self.sigma, self.rho))

        #vol surface

        self.strikes, self.ttm, self.expiries, self.df = vol_skeleton(strike_space, expiration_space,
                                                                    calc_date, spot_price)
        self.rf= rf
        self.dividend_rate = dividend_rate
        self.spot_price = spot_price
        self.v0 = ivar

        self.ircurve = set_rf(self.calc_date, self.rf)
        self.dividend = set_dividend(self.calc_date, self.dividend_rate)
        self.spot = set_spot(self.spot_price)
        process = self.initiate_process(self.kappa, self.theta, self.sigma, self.rho)
        model = self.mod_dict['model'](process)
        engine = self.mod_dict['engine'](model)

        eu_opt = [EU_option(self.calc_date, self.option_type, s, t)
                for s, t in zip(self.df['strikes'], self.df['ttm'])]
        [option.setPricingEngine(engine) for option in eu_opt]
        self.df['price'] = [option.NPV() for option in eu_opt]
        # self.df['Delta'] = [option.delta() for option in eu_opt]
    def initiate_process(self, kappa, theta, sigma, rho):
        return self.mod_dict['process'](self.ircurve, self.dividend, self.spot,
                self.v0, kappa, theta, sigma, rho)
# Helpers for Groupclass
def dt_to_ql(date):
    return ql.Date(date.day, date.month, date.year)
def ql_to_dt(Dateql):
    return dt.datetime(Dateql.year(), Dateql.month(), Dateql.dayOfMonth())

def ql_to_dt_settings_dict(Datesql, dictionary):
    if bool(dictionary):
        for dql in Datesql:
            helper = ql_to_dt(dql)
            dictionary[helper] = dictionary.pop(dql)

def dt_to_ql_settings_dict(dates, dictionary):
    if bool(dictionary):
        dates = sorted(dates)
        for date in dates:
            helper = dt_to_ql(date)
            dictionary[helper] = dictionary.pop(date)

def save_dict(dictionary, dict_name, path = data_dir):
    filehandle = open(path + '/{}.pkl'.format(dict_name), 'wb')
    pickle.dump(dictionary, filehandle)
    filehandle.close()

def load_dict(dict_name, path = data_dir):
    with open(path + '/{}.pkl'.format(dict_name), 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary

he_generation_bounds = [(0.5,10.), (0.05,0.8), (0.05,0.8), (-0.99,0.99)] #kappa,theta,sigma,rho
he_calibration_bounds = [(0.001,15.), (0.001,6.), (0.005,4.), (-0.999,0.999)]
he_mean_as_initial_point = [5., 1.5, 1., 0.]

def plot_vol_surf(Z, z_lab, main, string1 = '', string2 = '', W = None ,string3 = '',**kwargs):
    times = kwargs.get('maturities', maturities)
    money = kwargs.get('moneyness', moneyness)
    X = times; Y = money
    X, Y = np.meshgrid(X, Y)
    X = np.transpose(X); Y = np.transpose(Y)
    if Z.shape != (X.shape[0], Y.shape[1]):
        Z = Z.reshape((len(times), len(money)))
    if W is not None and W.shape != (X.shape[0], Y.shape[1]):
        W = W.reshape((len(times), len(money)))
    if W is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection = '3d')
        surf = ax.plot_surface(X,Y, Z, cmap = cm.coolwarm, linewidth = 0,  antialiased=True)
        fig.colorbar(surf, shrink = 0.6, aspect = 20, ax = ax)
        ax.set_xlabel('time to maturity'); ax.set_ylabel('moneyness'); ax.set_zlabel(z_lab)
        ax.text2D(0.06, 0.98, string1, transform=ax.transAxes)
        ax.text2D(0.06, 0.94, string2, transform=ax.transAxes)
        fig.suptitle(main)
        plt.show()
    else:
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(16)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        surf1 = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
        fig.colorbar(surf1, shrink=0.6, aspect=20, ax=ax)
        ax.set_xlabel('time to maturity (days)'); ax.set_ylabel('moneyness (%)'); ax.set_zlabel(z_lab)
        ax.text2D(0.06, 0.98, string1, transform=ax.transAxes)
        ax.text2D(0.06, 0.94, string2, transform=ax.transAxes)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf1 = ax.plot_surface(X, Y, Z, cmap=cm.Oranges, linewidth=0, antialiased=True, alpha=0.2)
        surf2 = ax.plot_surface(X, Y, W, cmap=cm.Blues, linewidth=0, antialiased=True)
        fig.colorbar(surf1, shrink=0.6, aspect=20, ax=ax)
        fig.colorbar(surf2, shrink=0.6, aspect=20, ax=ax)
        ax.text2D(0.15, 0.96, string3, transform=ax.transAxes)
        ax.set_xlabel('time to maturity (days)'); ax.set_ylabel('moneyness (%)'); ax.set_zlabel(z_lab)
        fig.suptitle(main)
        plt.show()

def approximation_function(x):
    return 0.5 + 0.5*np.sign(x)*np.sqrt(1. - np.exp(-2.*(x**2)/np.pi))
def StefanicaRadoicic(option_type, strike, premium, spot_price, ir_discount, dr_discount):
    fwd = spot_price * dr_discount / ir_discount
    ey = fwd / strike
    emy = strike / fwd
    y = np.log(ey)
    alpha = premium / (strike * ir_discount)

    if option_type == 1:
        # call
        R = 2. * alpha - ey +1.
    else:
        # put
        R = 2. * alpha - ey -1.
    pi_term = 2. /np.pi
    arg_exp_term = (1 - pi_term)*y
    R2 = R**2

    a = np.exp(arg_exp_term)
    A = (a - 1./a)**2
    b = np.exp(pi_term*y)
    B = 4.*(b + 1./b) - 2.*emy*(a + 1./a)*(ey**2 + 1. - R2)
    C = (emy**2) * (R2 - (ey - 1.)**2) * ((ey + 1.)**2 - R2)
    beta = 2.*C / (B + np.sqrt(B**2 + 4.*A*C))
    gamma = - np.pi/2.*np.log(beta)

    if y>=0.:
        if option_type==1: #call
            O0 = strike*ir_discount*(ey*approximation_function(np.sqrt(2.*y)) - 0.5)
        else:
            O0 = strike*ir_discount*(0.5 - ey*approximation_function(-np.sqrt(2.*y)))
        if premium <= O0:
            nu = np.sqrt(gamma+y) - np.sqrt(gamma-y)
        else:
            nu = np.sqrt(gamma+y) + np.sqrt(gamma-y)
    else:
        if option_type==1: #call
            O0 = strike*ir_discount*(0.5*ey - approximation_function(-np.sqrt(-2.*y)))
        else:
            O0 = strike*ir_discount*(approximation_function(np.sqrt(-2.*y)) - 0.5*ey)
        if premium <= O0:
            nu = np.sqrt(gamma-y) - np.sqrt(gamma+y)
        else:
            nu = np.sqrt(gamma+y) + np.sqrt(gamma-y)
    return nu

def Phi(x, nu):
        nu2 = nu **2
        abs_term = 2.*np.abs(x)
        return (nu2 - abs_term)/(nu2 + abs_term)
def N_plus(x,nu):
    return norm.cdf(x/nu + 0.5*nu)
def N_minus(x,nu):
    return np.exp(-x)*norm.cdf(x/nu - 0.5*nu)
def F(nu, x, c_star, omega):
    return c_star + N_minus(x,nu) + omega*N_plus(x,nu)
def G(nu, x, c_star, omega):
    argument = F(nu, x, c_star, omega)/(1.+omega)
    term = norm.ppf(argument)
    return term + np.sqrt(term**2 + 2.*np.abs(x))

def SOR_TS(option_type, strike, discount_ir, discount_dr, premium,
                     spot_price, guess, omega, tol, max_iter=20):
    assert (premium >=0.), 'Premium is zero bounded, must be positive.'

    fwd = spot_price * discount_dr / discount_ir
    x = np.log(fwd/strike)
    if option_type ==1:
        c = premium / (spot_price * discount_dr)
    else:
        c = premium / (spot_price * discount_dr) + 1. - strike / fwd
    if x > 0.:
        c = c * fwd/strike + 1. - fwd / strike
        assert (c >= 0.), 'Normalized premium is zero bounded, must be positive.'
        x = -x
    if not guess:
        guess = StefanicaRadoicic(option_type, strike, premium,
                                spot_price, discount_ir, discount_dr)
        assert (guess >= 0.), 'Initial guess is zero bounded, must be positive.'
        nIter = 0
        nu_k = nu_kp1 = guess
        diff = 1.
        while (np.abs(diff)> tol and nIter < max_iter):
            nu_k = nu_kp1
            alpha_k = (1.+omega)/(1.+Phi(x,nu_k))
            nu_kp1 = alpha_k*G(nu_k, x, c, omega) + (1.-alpha_k)*nu_k
            diff = nu_kp1 - nu_k
            nIter +=1
        return nu_kp1

def year_fraction(date, ttm):
    return [day_count.yearFraction(date, date + int(nd)) for nd in ttm]

class VolSurface:
    def __init__(self, option_type, spot_price = 100, rf = 0.01, dividend_rate = 0.0,
                 calc_date = starting_date, df = None):
        self.data = df

        if not 'ttm' in self.data.columns \
        or not 'strikes' in self.data.columns \
        or not 'price' in self.data.columns:
            raise RuntimeError('Some necessary data is missing')
        self.option_type = option_type
        self.calc_date = calc_date
        ql.Settings.instance().evaluationDate = self.calc_date

        self.ircurve = set_rf(self.calc_date, rf)
        self.dividend = set_dividend(self.calc_date, dividend_rate)
        self.spot = set_spot(spot_price)
        self.tol = 1.e-14

    def approximate_impvol(self, premium, strike, ttm):
        discount_ir = self.ircurve.discount(self.calc_date + int(ttm))
        discount_dr = self.dividend.discount(self.calc_date + int(ttm))
        sol = 0.
        try:
            sol = SOR_TS(self.option_type, strike, discount_ir, discount_dr,
                        premium, self.spot.value(),
                        guess = None, omega = 1., tol = self.tol)
        except (RuntimeWarning, AssertionError) as e:
            print(repr(e))
        yearly_ttm = day_count.yearFraction(self.calc_date, self.calc_date + int(ttm))
        sol = sol/np.sqrt(yearly_ttm)
        return sol
    def process_impvol(self):
        ivs = []
        for i in range(len(self.data['price'])):
            ivs.append(self.approximate_impvol(self.data['price'][i],
                                                self.data['strikes'][i],
                                                self.data['ttm'][i]))
            if ivs[i] == 0:
                raise ValueError('Zero implied volatility')
        return ivs

class HestonGroup:
    def __init__(self, heston_dict = he_analytic, first_date = starting_date,
                end_date = starting_date):
        self.mod_dict = heston_dict

        if ('model' not in self.mod_dict
            or 'process' not in self.mod_dict
            or 'engine' not in self.mod_dict
            or 'name' not in self.mod_dict
            or 'option_type' not in self.mod_dict):
            raise RuntimeError('Missing parameters in the dictionary')
        self.model_name = self.mod_dict['name'].replace("/", "").lower()

        self.option_type = self.mod_dict['option_type']
        self.dates = pd.date_range(ql_to_dt(first_date), ql_to_dt(end_date))

        self.dates_ql = [dt_to_ql(d) for d in self.dates]

        self.observables = {}
        self.he_parameters = {}
        self.he_df = {}
        self.he_ivs = {}

    # Methods for generating fake vol surface
    def generate_ir(self):
        return np.clip(np.random.beta(a = 1.01, b = 60), 0.001, 0.08)

    def generate_dr(self):
        return np.clip(np.random.beta(a = 1.01, b = 60), 0.001, 0.03)
    # Here, 100 = spot
    def generate_spot(self):
        return 100*(np.random.beta(a=8, b=8) + 1.0E-2)

    def generate_theta(self):
        return np.random.uniform(low=he_generation_bounds[1][0], high=he_generation_bounds[1][1])

    def generate_v0(self):
        return np.random.uniform(low = 0.001, high = 0.9)**2

    def generate_rho(self):
        return np.random.uniform(low=he_generation_bounds[3][0], high=he_generation_bounds[3][1])

    def generate_kappa(self):
        return np.random.uniform(low=he_generation_bounds[0][0], high=he_generation_bounds[0][1])

    def generate_sigma(self):
        return np.random.uniform(low=he_generation_bounds[2][0], high=he_generation_bounds[2][1])

    def Hestons(self, date, **kwargs):
        ir = kwargs.get('ir', self.generate_ir())
        dr = kwargs.get('dr', self.generate_dr())
        sp = kwargs.get('spot_price', self.generate_spot())
        v0 = kwargs.get('v0', self.generate_v0())
        rho = kwargs.get('rho', self.generate_rho())
        sigma = kwargs.get('sigma', self.generate_sigma())

        theta = kwargs.get('theta', self.generate_theta())

        kappa = kwargs.get('kappa', self.generate_kappa())
        while 2* kappa*theta <= sigma **2:
            theta = self.generate_theta()
            kappa = self.generate_kappa()

        return Heston(heston_dict = he_analytic, spot_price = sp, rf = ir,
                        dividend_rate = dr, calc_date = date,
                        expiration_space = maturities,
                        strike_space = moneyness, volvol = sigma,
                        corr = rho, eq_var = theta, ivar = v0,
                        mean_rev = kappa)

    def show_ivs(self, **kwargs):
        rd = self.dates_ql[0]
        hest_obj = self.Hestons(date = rd,**kwargs)
        obs_str = ['S0 =', 'ir =', 'dr =', 'v0 =']
        obs = [hest_obj.spot_price, hest_obj.rf, hest_obj.dividend_rate, hest_obj.v0]
        obs_str = ["{}{:.6}".format(o, str(v)) for o,v in zip(obs_str, obs)]
        params_str = [r'$\kappa = $', r'$\theta = $', r'$\sigma = $', r'$\rho = $']
        params = [hest_obj.kappa,hest_obj.theta, hest_obj.sigma, hest_obj.rho]
        params_str = ["{}{:.5}".format(o, str(v)) for o,v in zip(params_str, params)]

        ivs = VolSurface(option_type = hest_obj.option_type,
                        spot_price = hest_obj.spot_price,
                        rf = hest_obj.rf,
                        dividend_rate =  hest_obj.dividend_rate,
                        calc_date = hest_obj.calc_date,
                        df = hest_obj.df).process_impvol()
        plot_vol_surf(Z = np.array(ivs), z_lab = 'ivs', main = 'Heston Impvol surface',
                        string1 = ', '.join(o for o in obs_str),
                        string2 = ', '.join(o for o in params_str))

    def generate_historical_process_data(self, **kwargs):
        k = 0
        while k < len(self.dates_ql):
            datehelper = self.dates_ql[k]
            try:
                print('Date', datehelper)
                hest_obj = self.Hestons(datehelper)
                self.observables[datehelper] = np.array((hest_obj.spot_price, hest_obj.rf,
                                                        hest_obj.dividend_rate, hest_obj.v0))
                self.he_parameters[datehelper] = np.array((hest_obj.kappa,hest_obj.theta,
                                                          hest_obj.sigma,hest_obj.rho))
                print('Parameters: ',np.array((hest_obj.kappa,hest_obj.theta,
                                                          hest_obj.sigma,hest_obj.rho)))
                self.he_df[datehelper] = hest_obj.df
                ivs = VolSurface(option_type = hest_obj.option_type,
                                spot_price = hest_obj.spot_price,
                                rf = hest_obj.rf,
                                dividend_rate =  hest_obj.dividend_rate,
                                calc_date = hest_obj.calc_date,
                                df = hest_obj.df)
                self.he_ivs[datehelper] = ivs.process_impvol()
            except (ValueError) as e:
                print(e)
            else:
                k +=1
        if 'file_name' in kwargs and kwargs['file_name'] !='':
            self.model_name = self.model_name+'_'+kwargs['file_name']
        if 'save' in kwargs and kwargs['save'] == True:
            print('Saving historical data')

            ql_to_dt_settings_dict(self.dates_ql, self.observables)
            save_dict(self.observables, dict_name = 'he_observables')

            ql_to_dt_settings_dict(self.dates_ql, self.he_parameters)
            save_dict(self.he_parameters, dict_name = 'he_hist_parameters')

            ql_to_dt_settings_dict(self.dates_ql, self.he_df)
            save_dict(self.he_df, dict_name = 'he_df')

            ql_to_dt_settings_dict(self.dates_ql, self.he_ivs)
            save_dict(self.he_ivs, dict_name = 'he_ivs')

    def training_data_param_to_iv(self, no_samples, **kwargs):
        seed = kwargs.get('seed',0)
        print('Seed: %s'%seed)
        np.random.seed(seed)
        print('NN-training data are produced')
        x = []
        y = []

        fake_date = self.dates_ql[0]
        ql.Settings.instance().evaluationDate = fake_date
        i = 0
        while i < no_samples:
            try:
                hest_obj = None
                hest_obj = self.Hestons(fake_date)
                impvol_obj = VolSurface(option_type = hest_obj.option_type,
                                spot_price = hest_obj.spot_price,
                                rf = hest_obj.rf,
                                dividend_rate =  hest_obj.dividend_rate,
                                calc_date = hest_obj.calc_date,
                                df = hest_obj.df)
                ivs  = impvol_obj.process_impvol()
            except ValueError as e:
                print(hest_obj.parameters)
                print('Error: %s, sample %s'%(e,i+1)); print(' ')
            else:
                x.append([hest_obj.spot_price, hest_obj.rf, hest_obj.dividend_rate,
                        hest_obj.v0, hest_obj.kappa, hest_obj.theta,
                         hest_obj.sigma, hest_obj.rho])
                y.append(ivs)
                i += 1
        print('save', self.model_name)
        if 'file_name' in kwargs and kwargs['file_name'] !='': #needed for manual parallelization
             print('file name %s'%(self.model_name+'_'+kwargs['file_name']))
             self.model_name = self.model_name+'_'+kwargs['file_name']
        if 'save' in kwargs and kwargs['save'] == True:
            print('Saving data for training of NN')
            np.save(data_dir+'/'+self.model_name+'_train_he_find_iv_inputNN', x)
            np.save(data_dir+'/'+self.model_name+'_train_he_find_iv_outputNN', y)

        return (x,y)
bat_obj = Heston(heston_dict = he_analytic, spot_price = 100,
                rf = 0.01, dividend_rate = 0.0, ivar = 0.1,
                calc_date = starting_date, expiration_space = maturities,
                strike_space = moneyness,
                mean_rev = 5., eq_var = 0.2, volvol = 0.5, corr = -0.7)

predlist = list(predobj_obs.values())[0]
predobj_ivs = VolSurface(option_type = 1,
                spot_price = predlist[0],
                rf = predlist[1],
                dividend_rate =  predlist[2],
                calc_date = starting_date,
                df = list(predobj_df.values())[0]).process_impvol()
predivs_dict = {ql_to_dt(starting_date):predobj_ivs}
save_dict(predivs_dict, dict_name = 'predobj_ivs')


he_group = HestonGroup(first_date = starting_date,  end_date = starting_date)
# he_group.show_ivs()
start_date = dt.datetime(2019,1,1)
end_date = dt.datetime(2019,11,20)

f_date = starting_date
e_date = dt_to_ql(end_date)

# heston_processes = HestonGroup(first_date = f_date, end_date = e_date)
# heston_processes.generate_historical_process_data(save = True, seed = 1)
#
# heston_processes = HestonGroup()
#
#
#
# # heston_processes.training_data_param_to_iv(no_samples = 100000, seed = 1, save = True)
class MTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fun = None, inv_fun = None, validate = True,
                accept_sparse = False, pass_y = False):
        self.fun = fun
        self.inv_fun = inv_fun
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y

    def fit(self,X, y = None):
        if self.validate:
            check_array(X, self.accept_sparse)
        return self

    def transform(self, X, y = None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        if self.fun is None:
            return X
        return self.fun(X)
    def inverse_transform(self, X, y = None):
        if self.validate:
            X = check_array(X, self.accept_sparse)
        if self.inv_fun is None:
            return X
        return self.inv_fun(X)
def trnsf_he_param(y, transformation):
    y_len = y.shape[0]
    auxiliary_vec = (y[:,:7])
    auxiliary_vec = transformation(auxiliary_vec)
    y = np.concatenate((auxiliary_vec[:,:7], y[:,7].reshape(y_len, 1)), axis = 1)
    return y

def trnsf_he_prices(y, transformation):
    y = transformation(y)
    return y

def preprocess_he_param_NN(x, fun = he_analytic['transformation'],
                            inv_fun = he_analytic['inverse_transformation']):
    tr = MTransformer(fun = fun, inv_fun = inv_fun)
    scaler = MinMaxScaler(copy = True, feature_range = (-1, 1))
    pipe = Pipeline([('tr', tr), ('scaler', scaler)])
    x = trnsf_he_param(x, pipe.fit_transform)

    return x, pipe
def preprocess_premiums_or_vol(y):
    func = he_analytic['transformation']
    inv_func = he_analytic['inverse_transformation']
    tr = MTransformer(fun=func, inv_fun=inv_func)
    scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    pipe = Pipeline([('tr', tr), ('scaler', scaler)])
    try:
        y = pipe.fit_transform(y)
    except (TypeError, ValueError) as e:
        print(e)
        print('Max value: %s'%np.max(np.array(y)))
        print('Min value: %s'%np.min(np.array(y)))
    return y, pipe
scramble_seed = 1027
tot_s = 1.
val_s = 0.1
test_s = 0.1

def split_database(total_size, valid_size, test_size, total_sample):
    train_size = total_size - valid_size - test_size
    train_sample = int(round(total_sample*train_size))
    valid_sample = int(round(total_sample*valid_size))
    test_sample = int(round(total_sample*test_size))
    print(train_sample, valid_sample, test_sample)
    if train_sample < 1 or train_sample > total_sample or \
        valid_sample < 0 or valid_sample > total_sample or \
        test_sample < 0 or test_sample > total_sample:
        total_sample -= train_sample
        if total_sample - valid_sample < 0:
            valid_sample = 0
            test_sample = 0
        else:
            total_sample -= valid_sample
            if total_sample - test_sample < 0:
                test_sample = 0
    return train_sample, valid_sample, test_sample

def cleaning_data_eliminate_version(db, indexes=None):
    db = np.array(db)
    if indexes is None:
        indexes = (db<=0).sum(axis=1)
    db = db[indexes==0]
    return db, indexes
def get_trainingset_NN(file_name, fmodel,
                        positive_input = True, positive_output = True,
                        whiten_input = False, whiten_output = False,
                        seed = scramble_seed,total_size = tot_s,
                        valid_size = val_s, test_size = test_s):

    if positive_input and whiten_input:
        raise RuntimeError('Choose one preprocessing type for input')
    if positive_output and whiten_output:
        raise RuntimeError('Choose one preprocessing type for output')

    np.random.seed(seed)
    print('scrambling seed: %s' %seed)
    file_name = data_dir + '/' + file_name
    print('CWD: %s;  file name: %s'%(os.getcwd(), file_name))
    print('load', fmodel)
    x = np.load(file_name+'_train_'+fmodel+'_find_iv_inputNN.npy')
    y = np.load(file_name+'_train_'+fmodel+'_find_iv_outputNN.npy')

    y, indexes = cleaning_data_eliminate_version(db=y)

    print('Number of rejected samples: %s'%sum(indexes))

    x, _ = cleaning_data_eliminate_version(db=x, indexes=indexes)

    total_sample = y.shape[0]
    train_sample, valid_sample, test_sample = split_database(total_size, valid_size,
                                                             test_size, total_sample)
    if positive_input:
        print('==> Positive input')
        x, pipe_in = preprocess_he_param_NN(x)
    elif whiten_input:
        print('==> PCA & whitening in input')
        x, pipe_in = preprocess_prices_or_vola_pca(x)
    else:
        pipe_in = None

    if positive_output:
        print('==> Positive output')
        y, pipe_out = preprocess_premiums_or_vol(y)
    elif whiten_output:
        print('==> PCA & whitening in output')
        y, pipe_out = preprocess_prices_or_vola_pca(y)
    else:
        pipe_out = None

    index = np.arange(total_sample)
    np.random.shuffle(index)
    x_total, y_total = x[index], y[index]
    x_train, y_train = x_total, y_total

    if test_sample > 0:
        x_train, x_test, y_train, y_test =\
          train_test_split(x_train, y_train, test_size=test_sample, random_state=None)
    else:
        x_test = None
        y_test = None
    x_train, x_valid, y_train, y_valid =\
        train_test_split(x_train, y_train, test_size=valid_sample, random_state=None)

    return {'x_train': x_train, 'y_train': y_train,
            'x_valid': x_valid, 'y_valid': y_valid,
            'x_test': x_test, 'y_test': y_test,
            'preprocess_in': pipe_in, 'transform_out': pipe_out}
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=80, min_lr=9e-10, verbose=1)
earlyStopping = EarlyStopping(monitor='val_loss', patience=200)

def split_dataset(data):
    x_train = data['x_train']
    x_valid = data['x_valid']
    x_test = data['x_test']
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def proper_name(name):
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "_")
    name = name.replace("-", "_")
    return name

def reshape_fnn(x):
    if len(x[0].shape) == 1:
        p = np.concatenate(x)
        p.shape = (1, p.shape[0])
    else:
        p=x[None,:]
    return p

class NeuralNetwork(object):
    def __init__(self, model_name, model_callback, train_file_name, lr = 0.0005,
                loss = 'mean_squared_error', re_shape = reshape_fnn,
                prefix = '', postfix = '', fmodel = 'he', method = Nadam,
                checkPoint = True):
        self.model_name = model_name.lower()
        self.name = prefix + model_name
        self.postfix = postfix
        if self.postfix != '':
            self.postfix = '_'+self.postfix
        self.train_file_name = train_file_name
        self._fin_model = fmodel
        self.data = self.get_dta(fmodel = fmodel)
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.model = None
        self.history = None
        self.method = method
        self._transform = self.data['transform_out']
        self._model_callback = model_callback
        self.lr = lr
        self.loss = loss
        self._reshape = re_shape
        self._preprocessing = self.data['preprocess_in']
        self.checkPoint = checkPoint
        print('NeuralNetwork Class initialization finished.')

    def get_dta(self, fmodel):
        self.train_file_name = proper_name(self.train_file_name)
        self.train_file_name = self.train_file_name.replace('/', '_')
        print(self.train_file_name)
        return get_trainingset_NN(self.train_file_name, fmodel = fmodel)

    def file_names(self):
        file_name = proper_name(self.name) + '_nn' + self.postfix
        file_name = file_name.lower().replace('/', '_')
        return (data_dir + '/' + file_name, file_name)
    def ToFile(self):
        file_name, _ = self.file_names()
        if self.model is not None:
            json_file = file_name + '.json'
            json = self.model.to_json()
            open(json_file, 'w').write(json)
            if not self.checkPoint:
                print('Saving Weights...')
                h5file = file_name + '.h5'
                self.model.save_weights(h5file, overwrite = True)

    def FromFile(self):
        file_name,_ = self.file_names()
        json_file = file_name + '.json'
        if isfile(json_file):
            self.model = model_from_json(open(json_file).read())
            h5file = file_name + '.h5'
            self.model.load_weights(h5_file)
            print('Reading Neural Net from file and set learning rate to ', self.lr)
            method = self.method(lr = self.lr, clipnorm = 1.)
            self.model.compile(optimizer = method, loss = self.loss)
        else:
            self.model = None

    def GetState(self):
        print('Pickling the pickles')
        self.ToFile()
        model = self.model
        del self.__dict__['model']
        d = deepcopy(self.__dict__)
        self.model = model
        del d['data']
        del d['x_train']
        del d['x_valid']
        del d['x_test']
        del d['y_train']
        del d['y_valid']
        del d['y_test']
        return d

    def train(self, nb_epochs):
        print('Training NeuralNetwork with %s epochs.'%nb_epochs)
        if nb_epochs > 0:
            self.y_train = self.data['y_train']
            self.y_valid = self.data['y_valid']
            self.y_test = self.data['y_test']
            method = self.method(lr = self.lr, clipnorm = 1.)
            cp_file_name, simple_file_name = self.file_names()
            self.x_train, self.x_valid, self.x_test, self.model, self.history = \
                self._model_callback(self.data, method, self.loss, nb_epochs = nb_epochs, CP = self.checkPoint,
                                CP_name = cp_file_name, model_name = simple_file_name)
            if len(self.y_test) > 0:
                print(' '); print ('   -- NeuralNet on Test set --')
                print(self.model.evaluate(self.x_test, self.y_test,
                        batch_size = self.history['params']['batch_size']))
                print(' ')
    def fit(self, nb_epochs):
        if self.model is None:
            raise RuntimeError('Neural Net not yet instatiated')
        print('Fitting Neural Net...')
        batch_size = self.history['params']['batch_size']
        history2 = self.model.fit(self.x_train, self.y_train, batch_size = batch_size,
                                    nb_epochs = nb_epochs, verbose = 1,
                                    validation_data = (self.x_valid, self.y_valid))
        self.history = {'history': history2.history,
                        'params': history2.params}
    def predict(self, data):
        if self.model is None:
            raise RuntimeError('Neural Net not yet instantiated')
        if self._reshape is not None:
            data = self._reshape(data)
        if self._preprocessing is not None:
            data = trnsf_he_param(y = data, transformation = self._preprocessing.transform)
        y = self.model.predict(data)
        if self._transform is not None:
            y = trnsf_he_prices(y = y, transformation = self._transform.inverse_transform)
        return y
def NN_design(method, activation, exponent, init, layers, loss = 'mse',
                 dropout_first=None, dropout_middle=None, dropout_last=None,
                 neurons_first=None, neurons_last=None, weight_constraint=None,
                 dropout=None, tuning=False, **kwargs):
    c = weight_constraint

    if type(exponent)==str:
        exponent = eval(exponent)
    nb_unit = int(2**exponent)

    if dropout_first is None:
        dropout_first = dropout
    if dropout_middle is None:
        dropout_middle = dropout_first
    if dropout_last is None:
        dropout_last = dropout_middle

    act_idx = 1

    #model input
    inp = Input(shape = (neurons_first,))
    #model output
    ly = Dense(nb_unit, kernel_initializer=init,
               kernel_constraint=maxnorm(c, axis=0),
               use_bias=False)(inp)
    ly = BatchNormalization()(ly)

    act = copy(activation)
    act.name = act.name + '_'+str(act_idx)
    act_idx += 1
    ly = act(ly)
    ly = Dropout(dropout_first)(ly)

    for i in range(layers -1):

        middle = Dense(nb_unit, kernel_initializer=init,
                        kernel_constraint=maxnorm(c, axis=0),
                        use_bias=False)(ly)

        middle = BatchNormalization()(middle)
        act = copy(activation)
        act.name = act.name+'_'+str(act_idx)
        act_idx += 1
        middle = act(middle)
        middle = Dropout(dropout_middle)(middle)
        ly = add([ly, middle])
        act = copy(activation)
        act.name = act.name+'_'+str(act_idx)
        act_idx += 1
        ly = act(ly)


    ly = Dense(neurons_last, kernel_initializer=init,
                kernel_constraint=maxnorm(c, axis=0),
                use_bias=False)(ly)
    ly = BatchNormalization()(ly)
    act_idx += 1
    ly = Activation('linear')(ly)
    ly = Dropout(dropout_last)(ly)
    nn = Model(inputs = inp, outputs = ly)
    nn.compile(optimizer = method, loss = loss)

    return (nn, nb_unit, act_idx)
def fnn_model(data, method, loss = 'mse', exponent = 8, nb_epochs = 0,
                batch_size = 128, activation = 'tanh', layers = 4,
                init = 'he_uniform', dropout = 0.5, dropout_first = None,
                dropout_middle = None, dropout_last = None,
                neurons_first = None, neurons_middle = None,
                neurons_last = None, CP = True, CP_name = None,**kwargs):

    assert (isinstance(activation, string_types))

    if activation == 'elu':
        alpha = kwargs.get('alpha', 1.0)
        activation = ELU(alpha)
    else:
        activation = Activation(activation)
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(data)

    if not(neurons_first):
        neurons_first = x_train.shape[1]
    if not(neurons_last):
        neurons_last = y_train.shape[1]
    c = kwargs.get('c', None)

    nn, nb_unit, _ = NN_design(method, activation, exponent, init, layers, loss,
                                dropout_first, dropout_middle, dropout_last,
                                neurons_first, neurons_last,
                                weight_constraint = c, **kwargs)
    if nb_epochs >0:
        callbacks_list = [earlyStopping]
        callbacks_list.append(reduceLR)

        if CP:
            filepath = CP_name + '.h5'
            checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1,
                                        save_best_only = True, mode = 'min')
            callbacks_list.append(checkpoint)
            print('Callbacks:' + ', '.join([str(cb) for cb in callbacks_list]))

            history2 = nn.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epochs,
                                verbose = 2, callbacks = callbacks_list,
                                validation_data = (x_valid, y_valid))
            min_index, min_value = min(enumerate(history2.history['val_loss']),
                                        key = lambda p: p[1])
            print('Min losses - epoch %s and val_loss: %s, training loss: %s'%(
                    min_index+1, min_value, history2.history['loss'][min_index]))
            history = {'history': history2.history,
                        'params': history2.params}
    else:
        history = {'history' : [],
                    'params': []}
    return (x_train, x_valid, x_test, nn, history)

def generate_NN(exponent = 8, batch_size = 512, lr = 5e-5, layers = 6, loss = 'mse',
                activation = 'tanh', prefix = '', postfix = '', dropout = 0.5,
                dropout_first = None, dropout_middle = None, dropout_last = None,
                residual_cells = 0, **kwargs):

    init = kwargs.get('init', 'glorot_uniform')
    c = kwargs.get('c', None)
    fmodel = kwargs.get('fmodel', '')
    train_file_name = kwargs.get('train_file_name',
                            he_analytic['name'].lower())
    check_point = kwargs.get('check_point', True)

    callb = partial(fnn_model, exponent = exponent, batch_size = batch_size,
                    activation = activation, layers = layers, dropout = dropout,
                    dropout_first = dropout_first, dropout_middle = dropout_middle,
                    dropout_last = dropout_last, lr = lr, c = c, init = init)
    model = NeuralNetwork(he_analytic['name'].lower(), model_callback = callb, train_file_name = train_file_name,
                            fmodel = fmodel, lr = lr, loss = loss, re_shape = reshape_fnn, prefix = prefix,
                            postfix = postfix, checkPoint = check_point)
    return model

def write_model(model):
    model_file_name,_ = model.file_names()
    file_name = model_file_name + '.p'
    print('Saving Neural Net to file: %s' % file_name)

    dill.dump(model, open(file_name, 'wb'))
def read_model(file_name):
   file_name = file_name + '.p'
   print('Reading model from file: %s' % file_name)
   model = dill.load(open(file_name, 'rb'))
   return model

# model = generate_NN(exponent = 8, activation = 'elu',
#                     train_file_name = 'heston',
#                     layers = 4, lr = 5e-4,
#                     prefix = '', postfix = '',
#                     dropout_first = 0, dropout_middle = 0,
#                     dropout_last = 0, batch_size = 1024,
#                     fmodel = 'he',c = 5, check_point = True)
# model.train(nb_epochs = 250)
# write_model(model)
model = read_model('heston_nn')

def cost_function_calibr_with_NN(obj, predmodel, observables, l2=False, **kwargs):
    def cost_function(params):
        params = params.flatten().tolist()
        input_params = observables[:]
        input_params.extend(params)
        predicted_obj = (predmodel.predict(np.array(input_params))).ravel()

        diff = predicted_obj - obj
        if l2:
            return np.sum(diff**2)
        else:
            return np.sum(np.abs(diff))
    return cost_function

def heston_v0_as_param(observables, params):
    v0 = observables[3]
    parameters = params.tolist()
    parameters.insert(0, v0)
    return parameters

def calibration_through_NN(predmodel, fmodel, method):
    h_df = load_dict(dict_name = fmodel + '_df')
    dates = sorted(h_df.keys())
    h_observable = load_dict(dict_name = fmodel + '_observables')
    h_prices = {}

    for k in sorted(h_df.keys()):
        # eventuellt premium
        h_prices[k] = h_df[k]['price']
    h_variables = load_dict(dict_name = fmodel + '_ivs')

    h_params = load_dict(dict_name = fmodel + '_hist_parameters')
    cal_params = {}
    max_it = 100

    for date in dates:
        print(dt_to_ql(date))
        target_obj = np.array((h_variables[date]))
        observables = h_observable[date]
        observables = observables.flatten().tolist()
        cost_function = cost_function_calibr_with_NN(target_obj, predmodel, observables,
                                                    l2 = True)
        if method == 'slsqp':
            initial_guess = he_mean_as_initial_point
            sol = minimize(cost_function, initial_guess, bounds = he_calibration_bounds,
                            method = 'SLSQP', options = {'maxiter':max_it})
        elif method == 'diff_ev':
            sol = differential_evolution(func=cost_function,
                                         bounds=he_calibration_bounds,
                                         maxiter=max_it)
        cal_params[date] = sol.x
        print('Calibrated parameters: %s' %cal_params[date])
        print('Historical parameters: %s' %h_params[date])
        print('Final value from {} method:'.format(method), sol.fun )
        print('Iterations:', sol.nit);print(' ')
        parameters = heston_v0_as_param(observables, cal_params[date])

    dict_name = 'NN_calibrated_params_he'
    save_dict(dictionary = cal_params, dict_name = dict_name)

def predictive_calibration_NN(predmodel, fmodel, method):
    ivs = load_dict(dict_name = fmodel + '_ivs')
    observables = load_dict(dict_name = fmodel + '_observables')
    print(observables)
    df = load_dict(dict_name = fmodel + '_df')
    dates = sorted(df.keys())
    prices = {}
    for k in dates:
        prices[k] = df[k]['price']
    pred_cal_params = {}
    max_iter = 1000
    for date in dates:
        print(dt_to_ql(date))
        target = np.array((ivs[date]))
        observable = observables[date]
        observable = observable.flatten().tolist()

        cost_function = cost_function_calibr_with_NN(target, predmodel,
                                                    observable, l2 = True)
        if method == 'slsqp':
            initial_guess = he_mean_as_initial_point
            sol = minimize(cost_function, initial_guess,
                            bounds = he_calibration_bounds,
                            method = 'SLSQP', options = {'maxiter':max_iter})
        elif method == 'diff_ev':
            sol = differential_evolution(func = cost_function,
                                        bounds = he_calibration_bounds,
                                        maxiter = max_iter)

        pred_cal_params[date] = sol.x

        print('Predicted Calibrated parameters: %s' %pred_cal_params[date])
        print('Final value from {} method:'.format(method), sol.fun )
        print('Iterations:', sol.nit);print(' ')
        parameters = heston_v0_as_param(observable, pred_cal_params[date])
    dict_name = 'NN_predicted_calibrated_params'
    save_dict(dictionary = pred_cal_params, dict_name = dict_name)
# calibration_through_NN(predmodel = model, fmodel = 'he', method = 'diff_ev')
predictive_calibration_NN(predmodel = model, fmodel = 'predobj', method = 'diff_ev')

dispatch_names = {}
dispatch_names['he'] = 'Heston'

dispatch_string2 = {}
dispatch_string2['he'] = [r'$\kappa = $', r'$\theta = $', r'$\sigma = $', r'$\rho = $']

def params_to_str(obj, hpar, cpar, fmodel):
    obs_str = [r'$S_0 = $', 'ir = ', 'dr = ', r'$v_0 = $']
    obs = [obj[0],obj[1],obj[2],obj[3]]
    obs_str = ["{}{:.6}".format(o, str(v)) for o,v in zip(obs_str, obs)]

    par_str_h = ["{}{:.5}".format(o, str(v))
                            for o,v in zip(dispatch_string2[fmodel], hpar)]
    par_str_c = ["{}{:.5}".format(o, str(v))
                            for o,v in zip(dispatch_string2[fmodel], cpar)]
    return obs_str, par_str_h, par_str_c

def NN_calibration_plot(predmodel, date_index = 0):
    h_observable = load_dict(dict_name = 'he_observables')
    dates = sorted(h_observable.keys())
    date = dates[date_index]
    h_params = load_dict(dict_name = 'he_hist_parameters')
    h_ivs = load_dict(dict_name = 'he_ivs')
    cal_params = load_dict(dict_name = 'NN_calibrated_params_he')

    obj = h_observable[date]
    obj = obj.tolist()
    obs_str, hpar_str, cpar_str = params_to_str(obj, h_params[date].tolist(),
                                                cal_params[date].tolist(),'he')
    nn_input = np.concatenate((np.array(obj), cal_params[date]))

    W = predmodel.predict(nn_input)
    plot_vol_surf(Z = np.array(h_ivs[date]), z_lab = 'implied vols',
                main = dispatch_names['he'] + ' Calibrated implied vols with Neural Network',
                string1 = ', '.join(o for o in obs_str),
                string2 = 'Historical parameters: '+', '.join(o for o in hpar_str),
                W = W, string3 = 'Calibrated params: '+', '.join(o for o in cpar_str))
    return W
def NN_illustration_plot(predmodel, date_index = 0):
    h_observable = load_dict(dict_name = 'he_observables')
    dates = sorted(h_observable.keys())
    date = dates[date_index]
    h_params = load_dict(dict_name = 'he_hist_parameters')

    h_ivs = load_dict(dict_name = 'he_ivs')
    cal_params = load_dict(dict_name = 'NN_calibrated_params_he')

    obj = h_observable[date]
    obj = obj.tolist()
    obs_str, hpar_str, cpar_str = params_to_str(obj, h_params[date].tolist(),
                                                cal_params[date].tolist(),'he')
    nn_input = np.concatenate((np.array(obj), h_params[date]))

    W = predmodel.predict(nn_input)
    plot_vol_surf(Z = np.array(h_ivs[date]), z_lab = 'implied vols',
                main = dispatch_names['he'] + ' Calibrated implied vols with Neural Network',
                string1 = ', '.join(o for o in obs_str),
                string2 = 'Historical parameters: '+', '.join(o for o in hpar_str),
                W = W, string3 = 'Calibrated params: '+', '.join(o for o in cpar_str))

def NN_prediction_volplot(fmodel, predmodel, date_index = 0):
    h_observable = load_dict(dict_name = fmodel + '_observables')
    dates = sorted(h_observable.keys())
    date = dates[date_index]
    h_params = load_dict(dict_name = 'he_hist_parameters')

    h_ivs = load_dict(dict_name = fmodel + '_ivs')
    cal_params = load_dict(dict_name = 'NN_predicted_calibrated_params')
    obj = h_observable[date]
    obj = obj.tolist()
    obs_str, hpar_str, cpar_str = params_to_str(obj, h_params[date].tolist(),
                                                cal_params[date].tolist(),'he')
    nn_input = np.concatenate((np.array(obj), cal_params[date]))

    W = predmodel.predict(nn_input)
    # W = W.round(2)
    # print(W)
    # print('Market vol ===>', h_ivs[date])
    # print('__________________________')
    # print('Predicted Vol ===>', W)
    with open('predobj_predivs.pkl', 'wb') as f:
        pickle.dump(W, f)

    plot_vol_surf(Z = np.array(h_ivs[date]), z_lab = 'implied volatilities',
                main = 'S&P 500 index ' + dispatch_names['he'] + ' Calibrated implied vols with Neural Network',
                string1 = ', '.join(o for o in obs_str),
                string2 = '',
                W = W, string3 = 'Calibrated parameters: '+', '.join(o for o in cpar_str))
    return W



p_obs = load_dict(dict_name = 'predobj_observables')
predobs = list(p_obs.values())[0]
sp = predobs[0]
ir = predobs[1]
dr = predobs[2]
v0 = predobs[3]

predicted_params = load_dict(dict_name = 'NN_predicted_calibrated_params')
predicted_params = list(predicted_params.values())[0]
kappa = predicted_params[0]
theta = predicted_params[1]
sigma = predicted_params[2]
rho = predicted_params[3]

predicted_model = Heston(heston_dict = he_analytic, spot_price = sp, rf = ir,
                        dividend_rate = dr, calc_date = starting_date,
                        expiration_space = maturities,
                        strike_space = moneyness, volvol = sigma,
                        corr = rho, eq_var = theta, ivar = v0,
                        mean_rev = kappa)
process = predicted_model.initiate_process(kappa, theta, sigma, rho)

df = predicted_model.df
print(df.head())


h_obs = load_dict(dict_name = 'he_observables')
N = len(h_obs.keys())
NN_prediction_volplot(fmodel = 'predobj', predmodel = model, date_index  = 0)
# for i in range(N):
#     NN_calibration_plot(predmodel = model, date_index = i)
