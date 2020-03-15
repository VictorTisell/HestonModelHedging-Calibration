import inspect
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import QuantLib as ql
from itertools import product
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import math as m
import pickle
import dill
import types
dill._dill._reverse_typemap['ClassType'] = type
import warnings
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()


#### globals ####
seed = 2
np.random.seed(seed)
tf.set_random_seed(seed)
global_step = tf.Variable(0, trainable = False)
t0 = time.time()
current_dir = os.getcwd()
calibration_date = ql.Date(8,11,2019)
day_count = ql.Actual365Fixed()
calendar = ql.UnitedStates()
ql.Settings.instance().evaluationDate = calibration_date
### data helper functions
def get_pkl(filename, dict = True):
    extension = filename + '.pkl'
    with open(extension, 'rb') as f:
        output = pickle.load(f)
    if dict == True:
        output = list(output.values())[0]
    return output

def write_pkl(filename, object):
    extension = filename + '.pkl'
    with open(extension, 'wb') as f:
        pickle.dump(object, f)

#### data helper class ###
class data():
    def __init__(self):
        self.fmodel = 'predobj'
        self.observables_filename = self.fmodel + '_' + 'observables'
        self.observables = get_pkl(self.observables_filename)
        [self.S0, self.r, self.q, self.v0] = self.observables

        #### enable when file updated ####
        self.stochastic_model_parameters = get_pkl('NN_predicted_calibrated_params-2')
        # self.stochastic_model_parameters = [0.75509215,0.07400194, 0.33380185, -0.99450033]
        [self.kappa, self.theta, self.sigma, self.rho] = self.stochastic_model_parameters
        self.info = {'Market Observables': [self.S0, self.r, self.q, self.v0],
                            'Heston model parameters':[self.kappa, self.theta, self.sigma, self.rho]}
class parameters(data):
    def __init__(self):
        super().__init__()
        self.seed = 2
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.ttm = 31
        self.K = 3100

        self.dt = 1/365
        self.ttm_yrs = self.ttm * self.dt
        [self.S0, self.r, self.q, self.v0] = self.observables
        [self.kappa, self.theta, self.sigma, self.rho] = self.stochastic_model_parameters
        self.epsilon = 0.001
        self.alpha = 0.05
        self.lambda_riskaversion = 10
        ##### NN parameters #####
        self.nb_epochs = 31
        self.nb_samples = 500000
        self.batch_size = 258
        self.nb_batches = self.nb_samples // self.batch_size
        self.nb_test = 0.1 * self.nb_samples
        self.nb_validation = 0.1 * self.nb_samples
        self.learning_rate = 0.0005
        self.nb_features = 1
        self.nb_layers = 4
        self.training = 1
        self.decay = 0.999
        self.minimal_loss = 400
#### Financial helper functions ####
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

class stochastic(parameters):
    def __init__(self, calc_date = calibration_date, option_type = ql.Option.Call):
        super().__init__()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.spot = self.S0
        self.calc_date = calc_date
        self.spot = ql.SimpleQuote(self.spot)
        self.s0 = ql.QuoteHandle(self.spot)
        self.rf = set_rf(calc_date, self.r)
        self.ir = set_dividend(calc_date, self.q)
        self.process = ql.HestonProcess(self.rf, self.ir, self.s0, self.v0,
                                        self.kappa, self.theta, self.sigma, self.rho)
        self.engine = ql.AnalyticHestonEngine(ql.HestonModel(self.process))
        self.option = EU_option(calc_date, option_type, self.K, self.ttm)
        self.option.setPricingEngine(self.engine)

    def he_price(self):
        price = self.option.NPV()
        return price

    def he_delta(self, S, ttm):
        ql.Settings.instance().evaluation_date = self.calc_date
        X1 = S
        time_steps = int(ttm * 365)
        delta_hedge = []
        for k in range(0, X1.shape[0]):
            for i in range(0,X1.shape[2]):
                delta = [0]
                for t in range(1,time_steps-1):
                    s = float(X1[k,t,i])
                    valuation_date = self.calc_date + t
                    ql.Settings.instance().evaluation_date = valuation_date
                    self.spot.setValue(s)
                    u0 = self.spot.value()

                    h = s*0.01
                    self.spot.setValue(u0 + h)
                    dplus = self.option.NPV()
                    self.spot.setValue(u0 - h)
                    dminus = self.option.NPV()
                    dlta = (dplus - dminus)/(2*h)
                    delta.append(dlta)
                delta_hedge.append(delta)
        delta_hedge = np.stack(delta_hedge, axis = 0)
        delta_hedge = np.reshape(delta_hedge, [X1.shape[0], X1.shape[1]-1, X1.shape[2]])
        return delta_hedge

    def HeMM(self, ttm):
    	V = np.zeros(ttm)
    	S = np.zeros(ttm)

    	F = 0
    	S[0] = self.S0
    	V[0] = self.v0

    	for t in range(1,ttm):
    		Zv = np.random.normal(loc = 0, scale = 1)
    		Zs = self.rho*Zv + np.sqrt(1-self.rho**2)*np.random.normal(loc = 0, scale = 1)

    		num = 0.5*self.sigma**2 * V[t-1] * (1-np.exp(-2*self.kappa*self.dt)) / self.kappa
    		den = (np.exp(-self.kappa*self.dt)*V[t-1] + (1-np.exp(-self.kappa*self.dt))*self.theta)**2
    		Gam = np.log(1+num/den)
    		V[t] = (np.exp(-self.kappa*self.dt)*V[t-1] + (1-np.exp(-self.kappa*self.dt))*self.theta) * np.exp(-0.5*Gam**2 + Gam*Zv)
    		S[t] = S[t-1] * np.exp((self.r-self.q-V[t-1]/2)*self.dt + np.sqrt(V[t-1]*self.dt)*Zs)

    	return S, V

    def simulations(self, nb_sims):

        self.ttm = int(self.ttm)
        self.nb_features = int(self.nb_features)
        nb_sims = int(nb_sims)
        features = np.zeros(shape = (nb_sims, self.ttm, self.nb_features))
        labels = np.zeros((nb_sims))
        time_steps = self.ttm -1
        for i in range(nb_sims):
            price, vol = self.HeMM(ttm = time_steps)
            price = np.insert(price, 0, 0, axis = 0)
            features[i,:, 0] = price
            FinalPrices = features[i, time_steps, 0]
            labels[i] = np.maximum(FinalPrices - self.K, 0)
        labels = labels.reshape((nb_sims, self.nb_features))
        # print(np.mean(labels))
        return features, labels
def price(model):
    price = (model.he_price())
    price = np.ones((model.batch_size))*price
    q = tf.constant(price, dtype = tf.float32)
    return q

def price_test(model):
    price = (model.he_price())
    price = np.ones((int(model.nb_test)))*price
    q = tf.constant(price, dtype = tf.float32)
    return q

def price_valid(model):
    price = (model.he_price())
    price = np.ones((int(model.nb_validation)))*price
    q = tf.constant(price, dtype = tf.float32)
    return q

def generate_data(model):
    X, Y = model.simulations(nb_sims = model.nb_samples)
    return X, Y

def generate_test_data(model):
    X, Y = model.simulations(nb_sims = model.nb_test)
    return X, Y

def generate_validation_data(model):
    X, Y = model.simulations(nb_sims = model.nb_validation)
    return X, Y

def generate_batch(model, raw_data):
    raw_X, raw_y = raw_data
    data_X = np.zeros([model.batch_size, model.nb_features, model.ttm], dtype=np.float32)
    data_y = np.zeros([model.batch_size], dtype=np.float32)
    for i in range(model.nb_batches):
        data_X = (raw_X[model.batch_size * i:model.batch_size * (i + 1), :,:])
        data_y = raw_y[model.batch_size * i:model.batch_size * (i + 1)]
        yield (data_X, data_y)

def generate_epochs(model):
    for i in range(model.nb_epochs):
        yield generate_batch(model, model.dta)
def batch_norm(model, x, name):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        if model.training == 2:
            model.decay = 0
        elif model.training == 3:
            model.decay = model.nb_samples/(model.nb_samples + model.nb_test)
        param_shape = [x.get_shape()[-1]]
        batch_mean, batch_var = tf.nn.moments(x, [0], name = 'moments')
        pop_mean = tf.get_variable('moving_mean', param_shape,tf.float32,
                                    initializer=tf.constant_initializer(0.0),trainable=False)
        pop_var = tf.get_variable('moving_variance', param_shape,tf.float32,
                                    initializer=tf.constant_initializer(1.0),trainable=False)
        train_mean_op = tf.assign(pop_mean, pop_mean * model.decay + batch_mean * (1 - model.decay))
        train_var_op = tf.assign(pop_var, pop_var * model.decay + batch_var * (1 - model.decay))
        if model.training ==1:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, 0, 1, 1e-3)
        elif model.training == 2:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, 0, 1, 1e-3)
        elif model.training == 3:
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, pop_mean,pop_var, 0, 1, 1e-3)

def rnn_cell(model, rnn_input, state, name):
    with tf.variable_scope('rnn_cell' + str(name), reuse = True):
        input_ = tf.concat([rnn_input, state], axis = 1)
        W1 = tf.get_variable('W1', [model.nb_features*2, model.nb_features*2],
                            initializer = tf.random_normal_initializer(stddev = 0.1))
        b1 = tf.get_variable('b1', [model.nb_features*2],
                            initializer=tf.constant_initializer(0.0))
        W2 = tf.get_variable('W2', [model.nb_features*2, model.nb_features*2],
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', [model.nb_features*2],
                            initializer=tf.constant_initializer(0.0))
        W3 = tf.get_variable('W3', [model.nb_features*2, model.nb_features],
                            initializer=tf.random_normal_initializer(stddev=0.1))
        b3 = tf.get_variable('b3', [model.nb_features],
                              initializer=tf.constant_initializer(0.0))
        input = batch_norm(model, input_, 'layer_0')
        out1 = tf.matmul(input_, W1) + b1
        hidden_out1 = tf.nn.elu(out1)
        out2 = batch_norm(model, hidden_out1, 'layer_1')
        out2 = tf.matmul(out2, W2) + b2
        hidden_out2 = tf.nn.elu(out2)
        output =tf.matmul(hidden_out2, W3) + b3
        return output, W3, b3, W2, b2, W1, b1

class strategyfunctions(stochastic):
    def __init__(self):
        super().__init__()
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

    def strategy(self, outputs, X, y, nump = False, cumulative = False):
        price_changes = X[:, 1:, :] - X[:, :-1, :]
        helper0 = 0.05 * X[:,1:2,:]
        prices = X[:,1:,:]
        strategychangeshelper = outputs[:,1:,:] - outputs[:,:-1,:]
        strategychangeshelper = strategychangeshelper[:,1:,:]
        helper1 = outputs[:,1:2,:]
        helper2 = outputs[:,self.ttm -2:self.ttm-1,:]
        if cumulative:
            if nump:
                price_changes[:,0,:] = 0.05*price_changes[:,0,:]
                strategychanges = np.concatenate([helper1, strategychangeshelper, helper2], axis = 1)
                gains_of_trade = np.cumsum(np.sum(price_changes * outputs, axis = 2), axis = 1)
                transaction_costs = np.cumsum(np.sum(np.abs(prices)*np.abs(strategychanges), axis = 2), axis = 1)
            else:
                price_changes = price_changes[: , 1:, :]
                price_changes = tf.concat([helper0, price_changes], axis = 1)
                strategychanges = tf.concat([helper1, strategychangeshelper, helper2], axis = 1)
                gains_of_trade = tf.cumsum(tf.reduce_sum(price_changes * outputs, axis = 2), axis = 1)
                transaction_costs = tf.cumsum(tf.reduce_sum(np.abs(prices)*np.abs(strategychanges), axis = 2), axis = 1)
        else:
            if nump:
                price_changes[:,0,:] = 0.05*price_changes[:,0,:]
                strategychanges = np.concatenate([helper1, strategychangeshelper, helper2], axis = 1)
                gains_of_trade = np.sum(np.sum(price_changes * outputs, axis =1), axis =1)
                transaction_costs = np.sum(np.sum(np.abs(prices)*np.abs(strategychanges), axis =1), axis = 1)
            else:
                price_changes = price_changes[: , 1:, :]
                price_changes = tf.concat([helper0, price_changes], axis = 1)
                strategychanges = tf.concat([helper1, strategychangeshelper, helper2], axis = 1)
                gains_of_trade = tf.reduce_sum(tf.reduce_sum(tf.multiply(
                                price_changes, outputs), axis = 1), axis = 1)
                transaction_costs = tf.reduce_sum(tf.reduce_sum(tf.multiply(
                                    tf.abs(prices), tf.abs(strategychanges)), axis = 1), axis = 1)
        return gains_of_trade, transaction_costs

class RiskMeasures(strategyfunctions):
    def __init__(self):
        super().__init__()

    def quadratic_criterion(self, outputs, y, X):
        trading_strategy, transaction_costs = self.strategy(outputs, X, y)
        loss = tf.reduce_mean(tf.square(-tf.squeeze(y) + trading_strategy - self.epsilon * transaction_costs))
        return loss

    def expected_shortfall(self, outputs, y, X):
        trading_strategy, transaction_costs = self.strategy(outputs, X, y)
        loss,idx = tf.nn.top_k(-(-tf.squeeze(y) + trading_strategy - self.epsilon * transaction_costs),
                                tf.cast((1-self.alpha) * self.batch_size, tf.int32))
        CVaR = tf.reduce_mean(tf.abs(loss))
        return CVaR

    def entropy(self, outputs, y, X):
        trading_strategy, transaction_costs = self.strategy(outputs, X, y)
        loss =  tf.reduce_mean(tf.exp(tf.multiply(float(-self.lambda_riskaversion),
                                                (-tf.squeeze(y) + trading_strategy - self.epsilon * transaction_costs)*0.001)))
        entropic = tf.abs((1/self.lambda_riskaversion) * tf.log(loss))
        return entropic

    def worstcase(self, outputs, y, X):
        trading_strategy, transaction_costs = self.strategy(outputs, X, y)
        loss = tf.abs(tf.reduce_min(-tf.squeeze(y) + trading_strategy - self.epsilon * transaction_costs))
        return loss


    def hedgingloss(self, outputs, y, X, q = None, cumulative = False):
        if cumulative:
            trading_strategy, transaction_costs = self.strategy(outputs, X, y, nump = True, cumulative = True)

            if q is not None:
                one = np.ones(np.squeeze(trading_strategy).shape, dtype = np.float32)
                loss = trading_strategy - self.epsilon * transaction_costs + q*one
            else:
                loss = trading_strategy - self.epsilon * transaction_costs

        else:
            trading_strategy, transaction_costs = self.strategy(outputs, X, y, nump = True)
            if q is not None:
                one = np.ones(np.squeeze(y).shape, dtype = np.float32)
                loss = -np.squeeze(y) + trading_strategy + self.epsilon * transaction_costs + q*one
            else:
                loss = -np.squeeze(y) + trading_strategy + self.epsilon * transaction_costs
        return loss


model = stochastic()

class RNNModel(object):
    def __init__(self ,model ,neurons = [2*model.ttm,30,30,1],name = 'RNNmod'):
        # tf.reset_default_graph()
        self.activations = [tf.nn.elu, tf.nn.elu, tf.nn.elu, None]
        tf.set_random_seed(model.seed)
        np.random.seed(model.seed)
        self.X = tf.placeholder(tf.float32,
                            [model.batch_size, model.ttm, model.nb_features],
                            name = 'input_placeholder')
        self.y = tf.placeholder(tf.float32,[model.batch_size],
                            name = 'labels_placeholder')
        self.X_train = tf.placeholder(tf.float32,
                                    [model.nb_samples, model.ttm, model.nb_features],
                                    name = 'input_train_placeholder')
        self.X_valid = tf.placeholder(tf.float32,
                                    [model.nb_validation, model.ttm, model.nb_features],
                                    name='valid_input_placeholder')
        self.y_valid = tf.placeholder(tf.float32,[model.nb_validation],
                                    name='valid_labels_placeholder')
        self.X_test = tf.placeholder(tf.float32,
                                    [model.nb_test, model.ttm, model.nb_features],
                                    name='test_input_placeholder')
        self.y_test = tf.placeholder(tf.float32, [model.nb_test],
                                    name='test_labels_placeholder')

        self.initial_state = tf.zeros([model.batch_size, model.nb_features])
        for i in range(0, model.ttm -1):
            with tf.variable_scope('rnn_cell' + str(i)):
                W1 = tf.get_variable('W1', [model.nb_features*2, model.nb_features*2])
                b1 = tf.get_variable('b1', [model.nb_features*2],
                                    initializer = tf.constant_initializer(0.0))
                W2 = tf.get_variable('W2', [model.nb_features*2, model.nb_features*2])
                b2 = tf.get_variable('b2', [model.nb_features*2],
                                    initializer=tf.constant_initializer(0.0))
                W3 = tf.get_variable('W3', [model.nb_features*2, model.nb_features])
                b3 = tf.get_variable('b3', [model.nb_features],
                                    initializer=tf.constant_initializer(0.0))
        state = self.initial_state
        weights3 = []
        bias3 = []
        weights2 = []
        bias2 = []
        weights1 = []
        bias1 = []
        rnn_outputs = []
        for i in range(0, model.ttm -1):
            model.counter = i
            state, weight_matrix3, bias_vector3,  weight_matrix2, bias_vector2, weight_matrix1, bias_vector1  = rnn_cell(model, self.X[:, i, :], state, name=i)
            rnn_outputs.append(state)
            weights3.append(weight_matrix3)
            bias3.append(bias_vector3)
            weights2.append(weight_matrix2)
            bias2.append(bias_vector2)
            weights1.append(weight_matrix1)
            bias1.append(bias_vector1)
        rnn_outputs = tf.stack(rnn_outputs, axis = 1)
        weights3 = tf.stack(weights3, axis=1)
        bias3 = tf.stack(bias3, axis=1)
        weights2 = tf.stack(weights2, axis=1)
        bias2 = tf.stack(bias2, axis=1)
        weights1 = tf.stack(weights1, axis=1)
        bias1 = tf.stack(bias1, axis=1)
        self.hedging_weights = rnn_outputs
        self.test = rnn_outputs
        self.q = price(model)

        print('Running Neural Network')

        self.loss = RiskMeasures().expected_shortfall(self.hedging_weights, self.y, self.X)
        self.RM = 'ES005'
        # self.loss = RiskMeasures().entropy(self.hedging_weights, self.y, self.X)
        # self.RM = 'E10'
        # _, self.loss = RiskMeasures().entropy(self.hedging_weights, self.y, self.X)
        # self.RM = 'E10'

        opt = tf.train.AdamOptimizer(model.learning_rate)
        self.train_step = opt.minimize(self.loss)
        self.saver = tf.train.Saver()
        self.modelname = name

    def step(self, sess, batch_X, batch_y):
        input_feed = {self.X: batch_X,
                    self.y:np.squeeze(batch_y)}
        output_feed = [self.hedging_weights,
                        self.loss,
                        self.train_step,
                        self.q]
        outputs = sess.run(output_feed, input_feed)
        return (outputs[0], outputs[1], outputs[2], outputs[3])

    def sample_train(self, model):
        initial_state = tf.zeros([model.nb_samples, model.nb_features])
        hedging_weights = []
        state = initial_state
        for i in range(0, model.ttm -1):
            model.counter = i
            state,_,_,_,_,_,_ = rnn_cell(model, self.X_train[:,i,:], state, name = i)
            hedging_weights.append(state)
        hedging_weights = tf.stack(hedging_weights, axis = 1)
        return hedging_weights

    def sample(self, model):
        initial_state = tf.zeros([model.nb_test, model.nb_features])
        hedging_weights = []
        state = initial_state
        for i in range(0, model.ttm -1):
            model.counter = i
            state,_,_,_,_,_,_ = rnn_cell(model, self.X_test[:,i,:], state, name = i)
            hedging_weights.append(state)
        hedging_weights = tf.stack(hedging_weights, axis = 1)
        return hedging_weights

    def sample_validate(self, model):
        initial_state = tf.zeros([model.nb_validation, model.nb_features])
        hedging_weights = []
        state = initial_state
        for i in range(0, model.ttm -1):
            model.counter = i
            state,_,_,_,_,_,_ = rnn_cell(model, self.X_valid[:,i,:], state, name = i)
            hedging_weights.append(state)
        hedging_weights = tf.stack(hedging_weights, axis = 1)
        return hedging_weights

def build_model(sess, model):
    char_model = RNNModel(model)
    sess.run(tf.global_variables_initializer())
    return char_model

train_data = generate_data(model)
model.dta = train_data
X_valid, y_valid = generate_validation_data(model)
X_valid, y_valid = generate_validation_data(model)
X_test, y_test = generate_test_data(model)
earlytrainloss = np.zeros(model.nb_epochs)
earlyvalidateloss = np.zeros(model.nb_epochs)
metrics  = RiskMeasures()
with tf.Session() as sess:
    rnnmod = build_model(sess, model)
    saver = rnnmod.saver
    for idx, epoch in enumerate(generate_epochs(model)):
        training_losses = []
        for step, (input_X, input_y) in enumerate(epoch):
            predictions, total_loss, after_loss, q= rnnmod.step(sess, input_X, input_y)

            training_losses.append(total_loss)
        model_weights = model.he_delta(input_X, model.ttm_yrs)
        model_weights = np.concatenate([model_weights[:,:,0:model.nb_features],
                                        np.zeros([model.batch_size, model.ttm -1, 0])], axis = 2)

        model_hedge_loss = metrics.hedgingloss(model_weights, input_y, input_X,
                                                q = q, cumulative = False)
        deep_hedge_loss = metrics.hedgingloss(model_weights, input_y, input_X,
                                            q = None, cumulative = False)
        i = np.random.randint(0,(257),1)[0]


        model_hedge_loss_trail = metrics.hedgingloss(model_weights[i:i+1,:,:], input_y[i:i+1,:],
                                            input_X[i:i+1,:,:], q = q[i:i+1], cumulative = True)
        deep_hedge_loss_trail = metrics.hedgingloss(predictions[i:i+1,:,:], input_y[i:i+1,:],
                                            input_X[i:i+1,:,:], q = None, cumulative = True)
        print(deep_hedge_loss_trail)
        print(model_hedge_loss_trail)
        print('Model Starting Weight', model_weights[i:i+1,1,:])
        print('Deep Starting Weight', predictions[i:i+1,1,:])
        print('Heston Analytical Price:', q[i:i+1])
        print('Deep Hedge Price:', deep_hedge_loss_trail[0,0])
        print("Epoch %i, Loss: %.3f" % (idx,np.mean(training_losses)))
        model.training = 2
        train_batch = rnnmod.sample_train(model)
        X_train, y_train = train_data
        test_weights = sess.run([train_batch], {rnnmod.X_train: X_train})
        model.training = 3
        p = price_valid(model)
        validate_graph = rnnmod.sample_validate(model)
        loss = metrics.expected_shortfall(validate_graph, rnnmod.y_valid, rnnmod.X_valid)
        validate_loss, test_weights, certain_equivalent = sess.run([loss, validate_graph, rnnmod.q],
                                                        {rnnmod.X_valid:X_valid, rnnmod.y_valid: np.squeeze(y_valid)})
        print("Validate Loss: %.3f" % (validate_loss))
        model.training = 1
        if validate_loss < model.minimal_loss:
            save_path = saver.save(sess, './model/{}/ttm{}K{}'.format(rnnmod.RM, model.ttm, int(model.K)))
        earlytrainloss[idx] = np.mean(training_losses)
        earlyvalidateloss[idx] = validate_loss
    model.training = 2
    train_batch = rnnmod.sample_train(model)
    X_train, y_train = train_data
    test_weights = sess.run([train_batch], {rnnmod.X_train: X_train})
    model.training = 3
    p = price_test(model)
    test_graph = rnnmod.sample(model)
    loss = metrics.expected_shortfall(test_graph, rnnmod.y_test, rnnmod.X_test)
    test_loss, test_weights, certain_equivalent = sess.run([loss, test_graph, rnnmod.q],
                                                            {rnnmod.X_test: X_test, rnnmod.y_test: np.squeeze(y_test)})
    print("Loss on Test Data (unstopped): %.3f" % (test_loss))
    p_np = model.he_price()
    np.save('deep_hedge_pnl_{}_{}_{}.npy'.format(model.ttm, int(model.K),rnnmod.RM ), deep_hedge_loss_trail)

    root_logdir = './logs'
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")


    new_saver = tf.train.import_meta_graph('./model/{}/ttm{}K{}.meta'.format(rnnmod.RM, model.ttm, int(model.K)))
with tf.Session() as sess1:
    new_saver.restore(sess1, tf.train.latest_checkpoint('./model/{}/'.format(rnnmod.RM)))
    model.training = 2
    train_batch = rnnmod.sample_train(model)
    X_train, y_train = train_data
    test_weights = sess1.run([train_batch], {rnnmod.X_train: X_train})
    model.training = 3
    p = price_test(model)
    test_graph = rnnmod.sample(model)
    loss = metrics.expected_shortfall(test_graph, rnnmod.y_test, rnnmod.X_test)
    test_loss, test_weights, certain_equivalent = sess1.run([loss, test_graph, rnnmod.q],
                                                            {rnnmod.X_test: X_test, rnnmod.y_test: np.squeeze(y_test)})
    filewriter = tf.summary.FileWriter('{}/{}/run-{}'.format(root_logdir, rnnmod.RM, now), sess1.graph)
    print("Loss on Test Data (stopped): %.3f" % (test_loss))

t1 = time.time()
total_time=t1-t0
print("Runtime (in sec): %.3f" % (total_time))
np.save('deep_weights_ttm{}K{}_{}'.format(model.ttm, int(model.K), rnnmod.RM), test_weights)
model_hedge_weights = model.he_delta(X_test, model.ttm_yrs)

# model_hedge_weights = np.concatenate(model_hedge_weights[:,:,0:int(model.nb_features)],
#                                     np.zeros([int(model.nb_test), model.ttm - 1, 0]))

deep_hedge_loss = metrics.hedgingloss(test_weights, y_test, X_test)
p_np = stochastic().he_price()
model_hedge_loss = metrics.hedgingloss(model_hedge_weights, y_test, X_test,
                                        q = p_np)

np.save('stock_prices_ttm{}K{}_{}'.format(model.ttm, int(model.K), rnnmod.RM), X_test)
np.save('model_weights_ttm{}K{}_{}'.format(model.ttm, int(model.K), rnnmod.RM), model_hedge_weights)
df = pd.DataFrame(deep_hedge_loss)
df.to_csv('deep_hedge_loss_ttm{}K{}_{}'.format(model.ttm, int(model.K), rnnmod.RM))
df = pd.DataFrame(model_hedge_loss)
df.to_csv('model_hedge_loss_ttm{}K{}_{}'.format(model.ttm, int(model.K), rnnmod.RM))
print('neural network trained and stored')
