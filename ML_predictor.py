import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

import warnings

model_dict = {'ENet':[], 'SVR':[], 'DTree':[],
              'RF':[], 'AdaBoost':[], 'MLP':[]}


def gen_labels_factors(df):
    labels_df = pd.DataFrame(index=df.index[1:-1])
    factor_df = pd.DataFrame(index=df.index[1:-1])
    
    labels_df['interday_pct_change'] = df['adj_close_price'].pct_change().values[2:]
    
    factor_df['avg_volume_pct_change'] = df['average_volume_usd'].pct_change().values[1:-1]
    factor_df['volume_pct_change'] = df['volume_usd'].pct_change().values[1:-1]
    factor_df['intraday_pct_change'] = (df['adj_close_price'][1:-1] - df['adj_open_price'][1:-1])/df['adj_open_price'][1:-1]
    return labels_df, factor_df


def returns_to_prices(stock_price, returns_pred):
    return stock_price * (1 + returns_pred)


def ENet_predict(X_train, X_test, y_train, y_test):
    enet = ElasticNet()
    search_grid={'alpha':[0.1, 1, 10]}
    enet_search = GridSearchCV(estimator=enet, param_grid=search_grid, 
                               scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    enet_search.fit(X_train, y_train)
    para = enet_search.best_params_
    
    enet_regr = ElasticNet(alpha=para['alpha'])
    enet_regr.fit(X_test, y_test)
    y_pred_test = enet_regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return enet_regr, mse


def SVR_predict(X_train, X_test, y_train, y_test):
    svr = SVR()
    search_grid={'kernel':['rbf', 'poly'], 'C':[0.1, 1, 10], 'gamma':[0.1, 0.25]}
    svr_search = GridSearchCV(estimator=svr, param_grid=search_grid, 
                              scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    svr_search.fit(X_train, y_train)
    para = svr_search.best_params_
    
    svr_regr = SVR(kernel=para['kernel'], C=para['C'], gamma=para['gamma'])
    svr_regr.fit(X_test, y_test)
    y_pred_test = svr_regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return svr_regr, mse


def DTree_predict(X_train, X_test, y_train, y_test):
    dtree = DecisionTreeRegressor()
    search_grid={'max_depth':[3, 5, 7], 'min_samples_leaf':[0.1, 0.13, 0.15]}
    dt_search = GridSearchCV(estimator=dtree, param_grid=search_grid, 
                             scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    dt_search.fit(X_train, y_train)
    para = dt_search.best_params_
    
    dt_regr = DecisionTreeRegressor(max_depth=para['max_depth'], 
                                    min_samples_leaf=para['min_samples_leaf'], random_state=3)
    dt_regr.fit(X_test, y_test)
    y_pred_test = dt_regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return dt_regr, mse
    

def RF_predict(X_train, X_test, y_train, y_test):
    rf = RandomForestRegressor()
    search_grid={'n_estimators':[50, 100, 200]}
    rf_search = GridSearchCV(estimator=rf, param_grid=search_grid, 
                             scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    rf_search.fit(X_train, y_train)
    para = rf_search.best_params_
    
    rf_regr = RandomForestRegressor(n_estimators=para['n_estimators'], oob_score=True, random_state=100, )
    rf_regr.fit(X_test, y_test)
    y_pred_test = rf_regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return rf_regr, mse


def AdaBoost_predict(X_train, X_test, y_train, y_test):
    ada = AdaBoostRegressor()
    search_grid={'n_estimators':[50, 100],'learning_rate':[0.01,.1]}
    ada_search = GridSearchCV(estimator=ada, param_grid=search_grid,
                              scoring='neg_mean_squared_error',n_jobs=-1, cv=5)
    ada_search.fit(X_train, y_train)
    para = ada_search.best_params_
    
    ada_regr = AdaBoostRegressor(learning_rate=para['learning_rate'], n_estimators=para['n_estimators'], random_state=1)
    ada_regr.fit(X_test, y_test)
    y_pred_test = ada_regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return ada_regr, mse


def MLP_predict(X_train, X_test, y_train, y_test):
    mlp = MLPRegressor(hidden_layer_sizes=(3,4,3))
    search_grid={'activation':['relu', 'logistic', 'tanh']}
    mlp_search = GridSearchCV(estimator=mlp, param_grid=search_grid,
                              scoring='neg_mean_squared_error', n_jobs=-1, cv=5)
    mlp_search.fit(X_train, y_train)
    para = mlp_search.best_params_
    
    mlp_regr = MLPRegressor(hidden_layer_sizes=(3,4,3), activation=para['activation'], max_iter=500)
    mlp_regr.fit(X_test, y_test)
    y_pred_test = mlp_regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    return mlp_regr, mse


def predict(input_data, output_path=None):
    '''the main prediction function
    '''
    # validate the input
    dfin = input_data if isinstance(input_data, pd.DataFrame) else pd.read_csv(input_data)
    
    # Initialize the output dataframe
    dfout = pd.DataFrame(columns=['stockid', 'date', 'adj_close_price_pred_for_next_day'])
    dfout['stockid'] = dfin['stockid'].values
    dfout['date'] = dfin['date'].values
    
    # Predict for each stock
    stocks_list = list(dfin.groupby(['stockid']))
    for stock_info in stocks_list:
        stock_id, stock_data = stock_info[0], stock_info[1]
        labels, factors = gen_labels_factors(stock_data)
        factors[factors==np.inf] = np.nan
        factors = factors.fillna(method='pad')
        labels[labels==np.inf] = np.nan
        labels = labels.fillna(method='pad')
        
        X, y = factors.values, labels.values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
        
        # Find the best predictor
        for model_name in model_dict.keys():
            model_dict[model_name] = eval('{}_predict(X_train, X_test, y_train, y_test)'.format(model_name))
        best_model = min(model_dict.items(), key=lambda x: x[1][-1])
        best_model_name = best_model[0]
        print('stock {} -> best predictor used: {}'.format(stock_id, best_model_name))
        best_regr = best_model[1][0]
        
        best_regr.fit(X, y)
        returns_pred = best_regr.predict(X)
        # Adjust for the first day (no enough factor info) and the last day (no enough label info)
        returns_pred_adj = pd.Series([returns_pred[0]] + list(returns_pred) + [returns_pred[-1]])
        returns_pred_adj.index = stock_data.index
        # Get back to the price prediction
        price_pred = returns_to_prices(stock_data['adj_close_price'], returns_pred_adj)
        
        # Write to the output dataframe
        indices = stock_data.index
        dfout.loc[indices, 'adj_close_price_pred_for_next_day'] = price_pred
    
    # write output to csv
    dfout.to_csv(output_path, index=False)
    return dfout


def calc_error(input_data, output_data):
    dfin = input_data if isinstance(input_data, pd.DataFrame) else pd.read_csv(input_data)
    dfout = output_data if isinstance(output_data, pd.DataFrame) else pd.read_csv(output_data)
    dfm = pd.merge(dfin, dfout, on=['date', 'stockid'])
    dfm['adj_close_price_pred_for_next_day'] = dfm['adj_close_price_pred_for_next_day'].fillna(0)
    dfm['pred'] = dfm.groupby('stockid')['adj_close_price_pred_for_next_day'].shift(1)
    dfm['prev_adjclose'] = dfm.groupby('stockid')['adj_close_price'].shift(1)
    dfm['diff'] = (dfm['adj_close_price'] - dfm['pred']) / dfm['prev_adjclose']
    return (dfm['diff'] ** 2).mean()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    input_df = pd.read_csv('input_data.csv')
    
    print('Predicting')
    dfout = predict(input_df, output_path='output.csv')
    print('Evaluating ')
    error = calc_error(input_df, dfout)
    print("mean error = {error}".format(**locals()))

