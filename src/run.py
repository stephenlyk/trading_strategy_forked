#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
from strategy.moving_average import MovingAverage
from strategy.z_score import ZScore
from strategy.rsi import RSI
from strategy.roc import ROC
from strategy.percentile import Percentile
from strategy.min_max import MinMax
from strategy.robust import Robust
from joblib import Parallel, delayed
from optimization import Optimization

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Data
btc_price_df = pd.read_csv("../data/BTC-price_usd_close-NATIVE-2020_01_01-2024_07_12-24h.csv")[['timestamp', 'value']]
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])

# Factor
directory = '../data/factors'
price_factor_dict = {}
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        factor_df = pd.read_csv(file_path)[['timestamp', 'value']]

        factor_df.columns = ['Date', 'Target']
        factor_df["Date"] = pd.to_datetime(factor_df["Date"])
        price_factor_df = pd.merge(btc_price_df, factor_df, how='inner', on='Date')

        price_factor_dict[filename] = price_factor_df

#factor_df = pd.read_csv("../data/factors/" + filename)[['timestamp', 'value']]
#factor_df.columns = ['Date', 'Target']
#factor_df["Date"] = pd.to_datetime(factor_df["Date"])

#price_factor_df = pd.merge(btc_price_df, factor_df, how='inner', on='Date')

window_size_list = np.arange(2, 100, 5)
threshold_params = {
        'ZScore': np.round(np.linspace(0, 3, 42), 3),
        'MovingAverage': np.round(np.linspace(0, 5, 32), 3),
        'RSI': np.round(np.linspace(0.2, 0.8, 32), 4),
        'ROC': np.round(np.linspace(0, 0.8, 32), 4),
        'MinMax': np.round(np.linspace(0.1, 0.9, 42), 4),
        'Robust': np.round(np.linspace(0, 5, 32), 4),
        'Percentile': np.round(np.linspace(0.1, 0.9, 32), 4)
}
long_short_params = ['long', 'short', 'both']
condition_params = ['lower', 'higher']

overall_result = pd.DataFrame()

strategy_list = ['ZScore', 'MovingAverage', 'RSI', 'ROC', 'MinMax', 'Robust', 'Percentile']
def running_single_strategy(strategy, filename, price_factor_df, long_short, condition):

    test = Optimization(strategy, price_factor_df, window_size_list, threshold_params[strategy], target="Target", price='Price', long_short=long_short, condition=condition)
    test.run()

    optimization_params = pd.DataFrame({
        'Metric': filename,
        'Strategy': strategy,
        'Strategy Type': test.long_short,
        'Condition': test.condition
        }, index=[0])

    test_output = pd.concat([optimization_params, test.get_best().reset_index(drop=True)], axis=1)
    return test_output

for key in price_factor_dict:
    print(key)
    price_factor_df = price_factor_dict[key]
    for strategy in strategy_list:
        for long_short in long_short_params:
            for condition in condition_params:
                output = running_single_strategy(strategy, key, price_factor_df, long_short, condition)
                overall_result = pd.concat([overall_result, output], ignore_index=True)
    overall_result.to_csv("btc_factors_test.csv")

overall_result.to_csv("btc_factors_test.csv")
print(overall_result)


if __name__ == "__main__":

    print('done')
    #print(result.calmar)
    #print(result.sharpe)
    #result.result_df.to_csv("btc_liq_test.csv")
    #print(result.result_df)
    #result.plot_graph()

